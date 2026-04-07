"""
server/data.py — In-memory backend data store for the SupportTicket RL environment.

This module is the single source of truth for all episode data. Every order, customer,
product, and policy record is hard-coded here with no random generation. That determinism
is intentional: reproducible data means reproducible episodes, which is essential for
fair hackathon judging and reliable RL research.

The module exposes four dataclasses (Order, Customer, Product, Policy) and a BackendData
class that builds the canonical store at construction time. Each episode gets its own
deep-copied instance via BackendData.reset(), so mutations in one episode never bleed
into the next.

Design note: we deliberately keep this file free of FastAPI / Pydantic dependencies.
It is pure Python dataclasses so it can be imported by tests, the environment, and any
future tooling without pulling in the full server stack.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """Represents a single customer order in the e-commerce system.

    Args:
        order_id: Unique identifier, e.g. "ORD-1042".
        customer_id: Foreign key into the customers dict.
        product_id: Foreign key into the products dict.
        quantity: Number of units ordered.
        status: Lifecycle state — "processing", "shipped", or "delivered".
        total_amount: Amount actually charged to the customer (may differ from
            product.price * quantity if there was a billing error).
        order_date: ISO 8601 date string when the order was placed.
        delivery_date: ISO 8601 date string when the order was (or will be) delivered.
            None if not yet delivered.
        tracking_number: Carrier tracking code. None if not yet shipped.

    Returns:
        An Order instance with all fields populated.
    """

    order_id: str
    customer_id: str
    product_id: str
    quantity: int
    status: str
    total_amount: float
    order_date: str
    delivery_date: str | None
    tracking_number: str | None


@dataclass
class Customer:
    """Represents a registered customer account.

    Args:
        customer_id: Unique identifier, e.g. "CUST-001".
        name: Full display name.
        email: Contact email address.
        account_tier: Loyalty tier — "bronze", "silver", "gold", or "platinum".
            Tier affects the return window length (see return_policy).
        account_created: ISO 8601 date string when the account was opened.
        total_orders: Lifetime count of orders placed.
        lifetime_value: Total amount spent across all orders (USD).

    Returns:
        A Customer instance with all fields populated.
    """

    customer_id: str
    name: str
    email: str
    account_tier: str
    account_created: str
    total_orders: int
    lifetime_value: float


@dataclass
class Product:
    """Represents a product in the catalogue with current inventory.

    Args:
        product_id: Unique identifier, e.g. "PROD-001".
        name: Human-readable product name.
        category: Broad category string (e.g. "Electronics", "Accessories").
        price: Listed retail price in USD.
        warranty_days: Number of days the manufacturer warranty covers.
        is_returnable: Whether the product is eligible for return at all.
            Some items (e.g. opened software) are non-returnable by policy.
        stock_count: Current units available in the warehouse. Zero means out of stock.

    Returns:
        A Product instance with all fields populated.
    """

    product_id: str
    name: str
    category: str
    price: float
    warranty_days: int
    is_returnable: bool
    stock_count: int


@dataclass
class Policy:
    """Represents a named business policy with structured rules.

    Args:
        policy_key: Machine-readable identifier, e.g. "refund_policy".
        description: Human-readable summary of what the policy covers.
        rules: Flexible dict of rule key → value pairs consumed by the grader
            and reward logic. Using a dict rather than typed fields keeps the
            policy schema open for extension without changing the dataclass.

    Returns:
        A Policy instance with all fields populated.
    """

    policy_key: str
    description: str
    rules: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BackendData
# ---------------------------------------------------------------------------

class BackendData:
    """Canonical in-memory data store for one environment instance.

    At construction time, __init__ builds four dicts (orders, customers, products,
    policies) from hard-coded records. The reset() method returns a deep copy of
    self so each episode starts from a clean, identical state.

    Why a class rather than module-level dicts?
        Module-level dicts are singletons — mutating them in one episode would
        corrupt every subsequent episode. Wrapping the data in a class lets us
        hand each episode its own isolated copy via reset().

    Args:
        None — all data is hard-coded inside __init__.

    Returns:
        A BackendData instance ready for use as an episode's data store.
    """

    def __init__(self) -> None:
        """Build the canonical data store from hard-coded records.

        Every value here was chosen to support the three scenario narratives
        (easy: status inquiry, medium: damaged-item refund, hard: wrong item +
        billing error). Dates are anchored to a fictional "today" of 2024-01-15
        so the scenarios are always internally consistent regardless of when the
        code actually runs.
        """

        # ------------------------------------------------------------------
        # Orders
        # ------------------------------------------------------------------
        # We use a dict keyed by order_id for O(1) lookup. The four required
        # orders are listed first; additional orders follow for realism.

        self.orders: dict[str, Order] = {

            # ORD-1042 — easy scenario: customer wants to know where their order is.
            # Status is "shipped" (not yet delivered) so there's a tracking number
            # but no delivery_date yet. Delivery is expected in 2 days (2024-01-17).
            "ORD-1042": Order(
                order_id="ORD-1042",
                customer_id="CUST-001",
                product_id="PROD-001",
                quantity=1,
                status="shipped",
                total_amount=89.99,
                order_date="2024-01-10",
                delivery_date="2024-01-17",   # 2 days from "today" (2024-01-15)
                tracking_number="TRK-9821034",
            ),

            # ORD-2087 — medium scenario: customer received a damaged item and wants
            # a refund. Status is "delivered" so the return window is open. The
            # delivery was 5 days ago (2024-01-10), well within the 30-day window.
            "ORD-2087": Order(
                order_id="ORD-2087",
                customer_id="CUST-002",
                product_id="PROD-002",
                quantity=1,
                status="delivered",
                total_amount=149.99,
                order_date="2024-01-03",
                delivery_date="2024-01-10",   # delivered 5 days ago — within return window
                tracking_number="TRK-7654321",
            ),

            # ORD-3021 — hard scenario (part 1): customer ordered a Red Wireless Mouse
            # but received a Blue one instead. Wrong item received.
            "ORD-3021": Order(
                order_id="ORD-3021",
                customer_id="CUST-003",
                product_id="PROD-003",
                quantity=1,
                status="delivered",
                total_amount=45.00,
                order_date="2024-01-05",
                delivery_date="2024-01-12",
                tracking_number="TRK-1122334",
            ),

            # ORD-3022 — hard scenario (part 2): customer was overcharged by $15.
            # They should have paid $299.99 for the monitor but were billed $314.99.
            # The total_amount reflects what was actually charged (the erroneous amount).
            "ORD-3022": Order(
                order_id="ORD-3022",
                customer_id="CUST-003",
                product_id="PROD-004",
                quantity=1,
                status="delivered",
                total_amount=314.99,          # overcharged: correct price is $299.99
                order_date="2024-01-05",
                delivery_date="2024-01-12",
                tracking_number="TRK-5566778",
            ),

            # ORD-4001 — filler order: a processing order for CUST-004 to show the
            # system handles orders that haven't shipped yet (no tracking, no delivery).
            "ORD-4001": Order(
                order_id="ORD-4001",
                customer_id="CUST-004",
                product_id="PROD-005",
                quantity=2,
                status="processing",
                total_amount=59.98,
                order_date="2024-01-14",
                delivery_date=None,
                tracking_number=None,
            ),

            # ORD-4002 — filler order: a delivered order for CUST-005, no issues.
            "ORD-4002": Order(
                order_id="ORD-4002",
                customer_id="CUST-005",
                product_id="PROD-006",
                quantity=1,
                status="delivered",
                total_amount=199.99,
                order_date="2023-12-20",
                delivery_date="2023-12-27",
                tracking_number="TRK-9900112",
            ),

            # ORD-4003 — filler order: a second order for CUST-001 (gold tier) to
            # justify their high lifetime_value and total_orders count.
            "ORD-4003": Order(
                order_id="ORD-4003",
                customer_id="CUST-001",
                product_id="PROD-007",
                quantity=1,
                status="delivered",
                total_amount=349.99,
                order_date="2023-11-15",
                delivery_date="2023-11-22",
                tracking_number="TRK-3344556",
            ),

            # ORD-4004 — filler order: a shipped order for CUST-002 to give them
            # a second order in their history.
            "ORD-4004": Order(
                order_id="ORD-4004",
                customer_id="CUST-002",
                product_id="PROD-008",
                quantity=1,
                status="shipped",
                total_amount=79.99,
                order_date="2024-01-12",
                delivery_date="2024-01-18",
                tracking_number="TRK-6677889",
            ),
        }

        # ------------------------------------------------------------------
        # Customers
        # ------------------------------------------------------------------
        # Tiers are assigned to make the scenarios interesting:
        #   CUST-001 (gold)     — long-time customer, gets 30-day return window
        #   CUST-002 (silver)   — mid-tier, gets 21-day return window
        #   CUST-003 (bronze)   — newer customer, gets 14-day return window
        #   CUST-004 (bronze)   — very new account
        #   CUST-005 (platinum) — VIP customer, gets 45-day return window

        self.customers: dict[str, Customer] = {

            "CUST-001": Customer(
                customer_id="CUST-001",
                name="Alice Johnson",
                email="alice.johnson@example.com",
                account_tier="gold",
                account_created="2021-03-14",
                total_orders=12,
                lifetime_value=1247.85,
            ),

            "CUST-002": Customer(
                customer_id="CUST-002",
                name="Bob Martinez",
                email="bob.martinez@example.com",
                account_tier="silver",
                account_created="2022-07-22",
                total_orders=5,
                lifetime_value=489.94,
            ),

            # CUST-003 has two open issues (ORD-3021 and ORD-3022) — the hard scenario.
            # Bronze tier means a shorter return window, which the agent must check.
            "CUST-003": Customer(
                customer_id="CUST-003",
                name="Carol Lee",
                email="carol.lee@example.com",
                account_tier="bronze",
                account_created="2023-09-01",
                total_orders=3,
                lifetime_value=394.99,
            ),

            "CUST-004": Customer(
                customer_id="CUST-004",
                name="David Kim",
                email="david.kim@example.com",
                account_tier="bronze",
                account_created="2024-01-02",
                total_orders=1,
                lifetime_value=59.98,
            ),

            "CUST-005": Customer(
                customer_id="CUST-005",
                name="Eva Chen",
                email="eva.chen@example.com",
                account_tier="platinum",
                account_created="2019-06-10",
                total_orders=47,
                lifetime_value=8934.12,
            ),
        }

        # ------------------------------------------------------------------
        # Products
        # ------------------------------------------------------------------
        # PROD-001 through PROD-004 are required by the four scenario orders.
        # Additional products fill out the catalogue and test out-of-stock handling.
        # stock_count=0 on PROD-003 is intentional: the replacement scenario must
        # check inventory before issuing a replacement (and may find none available
        # for the exact item, prompting the agent to handle it gracefully).

        self.products: dict[str, Product] = {

            # PROD-001 — Wireless Headphones, ordered in ORD-1042 (easy scenario)
            "PROD-001": Product(
                product_id="PROD-001",
                name="Wireless Headphones",
                category="Electronics",
                price=89.99,
                warranty_days=365,
                is_returnable=True,
                stock_count=23,
            ),

            # PROD-002 — Mechanical Keyboard, ordered in ORD-2087 (medium scenario).
            # Arrived damaged, so the agent should issue a full refund.
            "PROD-002": Product(
                product_id="PROD-002",
                name="Mechanical Keyboard",
                category="Electronics",
                price=149.99,
                warranty_days=365,
                is_returnable=True,
                stock_count=8,
            ),

            # PROD-003 — Red Wireless Mouse, ordered in ORD-3021 (hard scenario).
            # Customer received a blue one instead. stock_count=0 means we cannot
            # send a replacement — the agent must handle this edge case.
            "PROD-003": Product(
                product_id="PROD-003",
                name="Red Wireless Mouse",
                category="Accessories",
                price=45.00,
                warranty_days=180,
                is_returnable=True,
                stock_count=0,              # out of stock — replacement not possible
            ),

            # PROD-004 — 4K Monitor, ordered in ORD-3022 (hard scenario).
            # Customer was overcharged $15; the agent should issue a $15 partial refund.
            "PROD-004": Product(
                product_id="PROD-004",
                name="4K Monitor",
                category="Electronics",
                price=299.99,
                warranty_days=730,          # 2-year warranty on monitors
                is_returnable=True,
                stock_count=4,
            ),

            # PROD-005 — USB-C Hub, ordered in ORD-4001 (filler)
            "PROD-005": Product(
                product_id="PROD-005",
                name="USB-C Hub",
                category="Accessories",
                price=29.99,
                warranty_days=180,
                is_returnable=True,
                stock_count=15,
            ),

            # PROD-006 — Smart Speaker, ordered in ORD-4002 (filler)
            "PROD-006": Product(
                product_id="PROD-006",
                name="Smart Speaker",
                category="Electronics",
                price=199.99,
                warranty_days=365,
                is_returnable=True,
                stock_count=11,
            ),

            # PROD-007 — Laptop Stand, ordered in ORD-4003 (filler)
            "PROD-007": Product(
                product_id="PROD-007",
                name="Laptop Stand",
                category="Accessories",
                price=349.99,
                warranty_days=365,
                is_returnable=True,
                stock_count=6,
            ),

            # PROD-008 — Webcam, ordered in ORD-4004 (filler).
            # Non-returnable once opened (common for webcams due to privacy concerns).
            "PROD-008": Product(
                product_id="PROD-008",
                name="HD Webcam",
                category="Electronics",
                price=79.99,
                warranty_days=365,
                is_returnable=False,        # non-returnable once opened
                stock_count=0,              # currently out of stock
            ),
        }

        # ------------------------------------------------------------------
        # Policies
        # ------------------------------------------------------------------
        # Policies are stored as structured dicts inside the Policy dataclass so
        # the grader and reward logic can inspect specific rule values without
        # parsing free text. Each key in `rules` is a machine-readable token.

        self.policies: dict[str, Policy] = {

            # refund_policy — governs when and how much we refund.
            # The 30-day window is measured from delivery_date, not order_date.
            # Damaged and wrong items get a full refund; billing errors get a
            # partial refund equal to the overcharge amount.
            "refund_policy": Policy(
                policy_key="refund_policy",
                description=(
                    "Customers may request a refund within 30 days of delivery. "
                    "Damaged or wrong items receive a full refund. "
                    "Billing errors receive a partial refund equal to the overcharge."
                ),
                rules={
                    "window_days": 30,
                    "full_refund_reasons": ["damaged", "wrong_item"],
                    "partial_refund_reasons": ["billing_error"],
                    "no_refund_reasons": ["changed_mind_after_window", "non_returnable"],
                },
            ),

            # replacement_policy — governs when we send a replacement item.
            # We must check inventory before committing to a replacement; if the
            # item is out of stock we should offer a refund instead.
            "replacement_policy": Policy(
                policy_key="replacement_policy",
                description=(
                    "Replacements are issued for wrong or defective items. "
                    "Inventory must be checked before confirming a replacement. "
                    "If the item is out of stock, offer a full refund instead."
                ),
                rules={
                    "eligible_reasons": ["wrong_item", "defective"],
                    "requires_inventory_check": True,
                    "fallback_if_no_stock": "full_refund",
                },
            ),

            # escalation_criteria — defines when a ticket must go to a human agent.
            # We keep the bar high (fraud or large refund) so the agent learns not
            # to escalate routine issues, which would incur a reward penalty.
            "escalation_criteria": Policy(
                policy_key="escalation_criteria",
                description=(
                    "Escalation to a senior agent is required only when fraud is "
                    "suspected or when the requested refund exceeds $500. "
                    "Routine refunds, replacements, and billing corrections do not "
                    "require escalation."
                ),
                rules={
                    "required_for": ["fraud_suspected"],
                    "refund_threshold_usd": 500.0,
                    "not_required_for": ["damaged_item", "wrong_item", "billing_error"],
                },
            ),

            # return_policy — return window varies by account tier.
            # Platinum customers get the longest window as a loyalty benefit.
            # The window is measured from delivery_date.
            "return_policy": Policy(
                policy_key="return_policy",
                description=(
                    "Return windows are tiered by account loyalty level. "
                    "Bronze: 14 days. Silver: 21 days. Gold: 30 days. Platinum: 45 days. "
                    "Items must be in original condition unless damaged in transit."
                ),
                rules={
                    "window_by_tier": {
                        "bronze": 14,
                        "silver": 21,
                        "gold": 30,
                        "platinum": 45,
                    },
                    "condition_required": "original_unless_damaged",
                },
            ),

            # shipping_policy — delivery time estimates by shipping method.
            # These are business-day estimates, not calendar days.
            "shipping_policy": Policy(
                policy_key="shipping_policy",
                description=(
                    "Standard shipping takes 3–5 business days. "
                    "Expedited shipping takes 1–2 business days. "
                    "Tracking numbers are assigned at the time of shipment."
                ),
                rules={
                    "standard_days_min": 3,
                    "standard_days_max": 5,
                    "expedited_days_min": 1,
                    "expedited_days_max": 2,
                    "tracking_assigned_on": "shipment",
                },
            ),
        }

        # Keep a reference to the canonical (unmodified) state so reset() can
        # always produce a clean copy even after mutations have been applied.
        # We store it as self._canonical pointing to self — the deep copy in
        # reset() will capture the full object graph at the time reset() is called.
        # This works because BackendData is only mutated through the apply_* methods,
        # never by replacing the top-level dicts themselves.
        self._canonical = self

    # ------------------------------------------------------------------
    # Episode isolation
    # ------------------------------------------------------------------

    def reset(self) -> "BackendData":
        """Return a deep copy of this BackendData instance for a new episode.

        Why deep copy instead of shallow copy?
            Shallow copy would share the nested dicts (orders, customers, products,
            policies) between the original and the copy. Any mutation — e.g.
            apply_refund() setting a "refund_applied" flag on an order, or
            apply_replacement() decrementing a product's stock_count — would then
            persist across episode resets, making episodes non-independent.
            Deep copy duplicates every nested object, guaranteeing that each
            episode starts from a pristine, identical state.

        Args:
            None

        Returns:
            A new BackendData instance with all data deep-copied from self.
        """
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_order(self, order_id: str) -> dict | None:
        """Return the order record for the given order ID, or None if not found.

        Args:
            order_id: The order identifier to look up, e.g. "ORD-1042".

        Returns:
            A dict representation of the Order dataclass, or None if the ID
            does not exist in the backend.
        """
        order = self.orders.get(order_id)
        if order is None:
            return None
        # Convert to dict so callers get a plain serialisable structure rather
        # than a dataclass reference. This also prevents callers from mutating
        # the internal Order object directly (they'd need to go through apply_*).
        return {
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "product_id": order.product_id,
            "quantity": order.quantity,
            "status": order.status,
            "total_amount": order.total_amount,
            "order_date": order.order_date,
            "delivery_date": order.delivery_date,
            "tracking_number": order.tracking_number,
        }

    def get_customer(self, customer_id: str) -> dict | None:
        """Return the customer record for the given customer ID, or None if not found.

        Args:
            customer_id: The customer identifier to look up, e.g. "CUST-001".

        Returns:
            A dict representation of the Customer dataclass, or None if the ID
            does not exist in the backend.
        """
        customer = self.customers.get(customer_id)
        if customer is None:
            return None
        return {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "email": customer.email,
            "account_tier": customer.account_tier,
            "account_created": customer.account_created,
            "total_orders": customer.total_orders,
            "lifetime_value": customer.lifetime_value,
        }

    def get_product(self, product_id: str) -> dict | None:
        """Return the product record for the given product ID, or None if not found.

        Args:
            product_id: The product identifier to look up, e.g. "PROD-001".

        Returns:
            A dict representation of the Product dataclass, or None if the ID
            does not exist in the backend.
        """
        product = self.products.get(product_id)
        if product is None:
            return None
        return {
            "product_id": product.product_id,
            "name": product.name,
            "category": product.category,
            "price": product.price,
            "warranty_days": product.warranty_days,
            "is_returnable": product.is_returnable,
            "stock_count": product.stock_count,
        }

    def get_policy(self, policy_type: str) -> dict | None:
        """Return the policy record for the given policy key, or None if not found.

        Args:
            policy_type: The policy key to look up, e.g. "refund_policy".

        Returns:
            A dict representation of the Policy dataclass, or None if the key
            does not exist in the backend.
        """
        policy = self.policies.get(policy_type)
        if policy is None:
            return None
        return {
            "policy_key": policy.policy_key,
            "description": policy.description,
            "rules": policy.rules,
        }

    def check_stock(self, product_id: str) -> int:
        """Return the current stock count for a product, or -1 if not found.

        Returning -1 (rather than None or raising) lets callers use a simple
        integer comparison without needing to handle a None case separately.
        -1 is an unambiguous sentinel because real stock counts are always >= 0.

        Args:
            product_id: The product identifier to check, e.g. "PROD-003".

        Returns:
            The current stock_count integer, or -1 if the product ID is unknown.
        """
        product = self.products.get(product_id)
        if product is None:
            return -1
        return product.stock_count

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def apply_refund(self, order_id: str, amount: float) -> bool:
        """Mark a refund as applied on the specified order.

        This method records that a refund has been issued so the grader and
        reward logic can verify the correct action was taken. It does not
        actually move money — this is a simulation.

        Args:
            order_id: The order to apply the refund to.
            amount: The refund amount in USD.

        Returns:
            True if the order exists and the refund was recorded, False if the
            order ID is not found in the backend.
        """
        order = self.orders.get(order_id)
        if order is None:
            return False

        # We store refund metadata directly on the Order object using dynamic
        # attributes rather than adding fields to the dataclass. This keeps the
        # dataclass schema clean while still allowing the grader to inspect
        # whether a refund was applied and for how much.
        # Why not add a field to the dataclass? Because refund state is episode-
        # specific (it gets reset with each deep copy) and adding it to the
        # dataclass would require a default value, cluttering the constructor.
        order.refund_applied = True          # type: ignore[attr-defined]
        order.refund_amount = amount         # type: ignore[attr-defined]
        return True

    def apply_replacement(self, order_id: str, product_id: str) -> bool:
        """Record a replacement and decrement the product's stock count.

        Before calling this method, callers should verify stock is available via
        check_stock(). This method will still decrement even if stock is 0,
        because the environment layer is responsible for the pre-check; this
        method just executes the mutation.

        Args:
            order_id: The order for which a replacement is being sent.
            product_id: The product ID of the replacement item.

        Returns:
            True if both the order and product exist and the replacement was
            recorded, False if either ID is not found.
        """
        order = self.orders.get(order_id)
        product = self.products.get(product_id)
        if order is None or product is None:
            return False

        # Decrement stock to reflect that one unit has been allocated for the
        # replacement shipment. We clamp at 0 to avoid negative stock counts,
        # which would be nonsensical in a real warehouse system.
        product.stock_count = max(0, product.stock_count - 1)

        # Record replacement metadata on the order for grader inspection.
        order.replacement_applied = True     # type: ignore[attr-defined]
        order.replacement_product_id = product_id  # type: ignore[attr-defined]
        return True
