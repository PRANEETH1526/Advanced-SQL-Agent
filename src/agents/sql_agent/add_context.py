from agents.vectorstore import create_collection, get_collection, insert_data, sear

if __name__ == "__main__":
    # Create a collection
    create_collection()

    collection = get_collection()

    doc1 = """
    Tags: component, component ID, ordered, purchased, date, year, price, total, quantity, category, type, 

#### Tables:
1. **purchase_orders**  
   - **Description**: Stores purchase order details covering vendor info, authorization, shipping, and financial terms.  
   - **COMMENT Section**:  
     - Tracks order revisions, approvals (`auth_id`, `auth_date`, etc.), and associated notes (`amended_notes`, `order_comments`, `delivery_comments`, `notes`).  
     - Includes vendor contact details (`vendor_name`, `attention`, `address`, `vendor_email`, `vendor_fax`), shipping methods (`shipping`, `shipping_extra`), and incoterms.  
     - Contains fields for document attachments and invoicing (`attachments`, `po_email_attachments`, `po_invoicing_check`) plus internal payment terms and status flags (`po_status`, `po_closed`). 
     - **Joins**:  
       - `purchase_order_lines.po_id = purchase_orders.id` (Each line belongs to a purchase order).  
       - `comp_lot.purchase_order = purchase_orders.po_no` (Each lot was purchased on a purchase order).  
       - `purchase_orders.vendor_id = Company.CompanyID`.

2. **purchase_order_lines**  
   - **Description**: Tracks individual line items for purchase orders, providing detailed insight into each order’s components and pricing.  
   - **COMMENT Section**:  
     - The primary key `id` uniquely identifies each line, and `po_id` links the line to its parent order, while `lot_no` optionally ties the line to a specific inventory lot.  
     - Key fields include `qty_type` (unit of measure), `vendor_code` (vendor’s item identifier), `description`, `price`, and `qty`.  
     - Shipment dates (`drop_date`, `po_line_edd`) and GST applicability (`gst_applicable_line`) define delivery terms.  
     - Versioning is tracked via `version_timestamp`, `version_editor`, and `version_number`.  
     - Order status is managed with `cancelled` and `received` (non-lot items) flags, while authorization controls (`non_preferred_auth`, `price_increase_auth`, `auth_time_price`) and supplementary notes (`pol_purchasing_notes`, `pol_internal_notes`, `pol_freight_tracking`, `pol_risk`, `pol_inspection_instructions`, `pol_supplier_requirements`, `pol_quote_ref`) support internal controls and risk management.  
     - **Joins**:  
       - `purchase_order_lines.lot_no = comp_lot.lot_no` (If a line references a specific inventory lot).

3. **component**  
   - **Description**: Stores component data representing groups of parts conforming to specific specifications. Each component is uniquely identified (e.g., ID6334) and includes a descriptive field detailing its purpose and application.  
   - **COMMENT Section**:  
     - Components are categorized (e.g., capacitor, resistor, end product, subassembly) via the `component.cat_no` field, which links to `category.ID` for added context.  
     - Functionally equivalent manufacturer parts are grouped under one component ID to support flexible sourcing.  
     - Sourcing details (including vendor and manufacturer part numbers) are managed in the `comp_src` table, while lot tracking—with identifiers formatted as DDMMYY-X (and appended suffixes for split lots)—is handled in `comp_lot`.  
     - In-house production details and BOMs (e.g., PCB revisions and codes) are captured in related tables to aid production planning and quality control.  
     - **Joins**:  
       - `component.ID = comp_src.ID` (Each sourcing record is linked to a component via its ID).  
       - `category.cat_no = category.ID` (Each component is of a particular category).

4. **comp_lot**  
   - **Description**: Tracks detailed information for component lots, providing batch-level visibility into inventory and sourcing.  
   - **COMMENT Section**:  
     - The primary key `lot_id` uniquely identifies each lot, while `lot_no` serves as a human-readable identifier formatted as DDMMYY-X, with optional suffixes for split lots.  
     - Key fields include `lot_date` (creation date), `on_order` (flag to indicate that it is not yet received), `price` (per part in AUD), `on_hand` (current inventory), and `purchase_order` with `purchased_amount` (acquisition details).  
     - `edd` is the expected delivery date while the lot is on order, after which it represents the receive date.  
     - The `source_ID` field links each lot to its sourcing record in `comp_src`, ensuring traceability.  
     - Additional fields capture storage specifics (`location_ID`, `shelf`, `position`, `box_id`), versioning (`version_timestamp`, `version_editor`, `version_number`), quality control data (`qc_notes`, `qc_pass`, `qc_who`, `qc_date`), pricing, and supplementary notes, supporting comprehensive inventory and quality management.  
     - **Joins**:  
       - `comp_src.source_ID = comp_lot.source_ID` (Each sourcing record links to a specific lot via `source_ID`).  
       - `purchase_order_lines.lot_no = comp_lot.lot_no` (If a line references a specific inventory lot).  
       - `comp_lot.purchase_order = purchase_orders.pono` (Each lot was purchased on a particular PO).  

5. **comp_src**
   - **Description**: Stores sourcing details for components, linking each component’s `ID` to its associated vendor and manufacturer data. Records include `vendor_part_no` and `manufacturer_part_no`, ensuring that parts from different manufacturers but functionally equivalent are grouped under a single component. The `manufacturer` field and `vendor_ID` reference corresponding `Company` IDs, facilitating integration with vendor data. The `source_ID` field connects sourcing records to specific component lots in `comp_lot`, enhancing traceability.
   - **COMMENT Section**:
     - Join on `component.ID = comp_src.ID` (each sourcing record is linked to a component via its ID).
     - Join on `comp_src.source_ID = comp_lot.source_ID` (each sourcing record links to a specific lot via `source_ID`).
     - Join on `comp_src.manufacturer = Company.CompanyID` (each sourcing record is linked to a Company as a manufacturer).
     - Join on `comp_src.vendor_ID = Company.CompanyID` (each sourcing record is linked to a Company as a vendor).

6. **category**
   - **Description**: Category table is used to categorize components into different categories - e.g., integrated circuit, capacitor, resistor, metalwork.
   - **COMMENT Section**:
     - Key field is `ID`.
     - `category.category` field is a text field indicating the component category.
     - Joins to `component` table on `component.cat_no = category.ID`.

---

#### Key Relationships:
1. **purchase_orders → purchase_order_lines**:  
   - `purchase_order_lines.po_id = purchase_orders.id` (Each line belongs to a purchase order).  

2. **purchase_order_lines → component**:  
   - `purchase_order_lines.lot_no = comp_lot.lot_no` (If a line references a specific inventory lot).  
   - `component.ID` uniquely identifies the component (e.g., 6334).  

3. **purchase_order_lines → comp_lot**:  
   - `purchase_order_lines.lot_no = comp_lot.lot_no` (If a line references a specific inventory lot).  

4. **comp_lot → purchase_orders**:  
   - `comp_lot.purchase_order = purchase_orders.pono` (Each lot was purchased on a particular purchase order).  

5. **comp_lot → comp_src**:
   - `comp_lot.source_ID = comp_src.source_ID` (Each lot is linked to a sourcing record for traceability).

6. **comp_lot → purchase_order_lines**:
   - `purchase_order_lines.lot_no = comp_lot.lot_no` (Each purchase order line may reference a specific inventory lot).

7. **comp_src → component**:
   - `comp_src.ID = component.ID` (Each sourcing record is linked to a specific component).

8. **component → category**:
   - Join Condition: `component.cat_no = category.ID`
   - Description: Links each component to its respective category, providing context for classification (e.g., capacitor, resistor, etc.).


---

#### Relevant Fields for the User Question:
1. **purchase_orders**:  
   - `id` (INTEGER): Primary key of the purchase orders table.  
   - `date_issued` (DATE): Date the purchase order was issued.  

2. **purchase_order_lines**:  
   - `po_id` (INTEGER): Links the line to its parent purchase order.  
   - `qty` (FLOAT): Quantity ordered.  

3. **component**:  
   - `ID` (INTEGER): Primary key and text identifier of the component (e.g., 6334).  

4. **comp_lot**:  
   - `lot_no` (VARCHAR): Key field linking to `purchase_order_lines.lot_no`.  
   - `purchase_order` (INTEGER): Links the lot to the purchase order number.  
   - `purchased_amount` (FLOAT): Quantity purchased in the lot.  
   - `lot_date` (DATE): Date the lot was created. 
   - `price` (FLOAT): Unit price, converted to AUD.
   - `source_ID` (INTEGER): Links the lot to its sourcing record in `comp_src`.
   - `edd` (DATE): Expected delivery date (if on order) or the date the lot was received.
   - `on_order` (TINYINT): Flag indicating if the lot is still on order (1 = on order).

5.- **From `comp_src`**:
  - `source_ID` (INTEGER): Primary key for the sourcing record.
  - `ID` (INTEGER): Links the sourcing record to a specific component.
  - `vendor_ID` (INTEGER): Links the sourcing record to a vendor in the `Company` table.
  - `manufacturer` (INTEGER): Links the sourcing record to a manufacturer in the `Company` table.
  - `vendor_part_no` (VARCHAR): Vendor's identifier for the part.
  - `manufacturer_part_no` (VARCHAR): Manufacturer's identifier for the part.

6. **From `category`**:
   - `ID`: Primary key and unique identifier for each category.
   - `category`: Text field indicating the name of the component category.

#### SQL Example:

Capacitators ordered in 2023:

SELECT SUM(qty) AS 'a) TOTAL EVER', SUM(IF(po.date_issued BETWEEN '2023-01-01' AND '2023-12-31',qty,0)) AS 'TOTAL in 2023'

FROM purchase_orders po

JOIN purchase_order_lines plx ON po.id = plx.po_id

JOIN comp_lot cl ON cl.lot_no = plx.lot_no

JOIN comp_src cs ON cs.source_ID = cl.source_ID

JOIN component c ON cs.id = c.id

JOIN category cat ON c.cat_no = cat.ID

AND cat.category = 'capacitor'
    """

    doc2 = """
    Tags: dockets, company, delivery

#### Tables:
**intelli_ddockets**  
- **Description**: This table stores delivery docket details, including docket numbers, shipping information, client details, and associated metadata.  
- **COMMENT Section**:  
  - The `intelli_ddockets` table is primarily used to track delivery dockets.  
  - Key fields include `ddocket_id` (primary key), `ddocket_no` (delivery docket number), and `shipping_date` (date of delivery).  
  - The table also contains fields for client information (`client_name`, `address`, `address_country`), shipping details (`shipped_via`, `shipped_via_other`), and additional metadata such as `order_date`, `comments`, and `prepared_by`.  
  - Relationships:  
    - The `company_id` field links to the `Company` table, which provides details about the company associated with the delivery docket.  
    - The `wor_id` field may link to a work order or related table for tracking purposes.  

---

#### Key Relationships:
1. **intelli_ddockets → Company**:  
   - Join Condition: `intelli_ddockets.company_id = Company.CompanyID`  
   - Purpose: To retrieve company details (e.g., name, address) associated with the delivery docket.  
---

#### Relevant Fields for the User Question:
- **intelli_ddockets**:  
  - `ddocket_no`: The delivery docket number.  
  - `shipping_date`: The corresponding date of the delivery docket.  
"""

    doc3 = """
    Tags: purchase order, POs, date, shipped


#### Tables:
1. **purchase_order_lines**  
   - **Description:** Tracks individual line items for purchase orders, providing detailed insight into each order’s components and pricing. The primary key `id` uniquely identifies each line, and `po_id` links the line to its parent order, while `lot_no` optionally ties the line to a specific inventory lot.  
   - **COMMENT Section:**  
     - Key fields include `qty_type` (unit of measure), `vendor_code` (vendor’s item identifier), `description`, `price`, and `qty`.  
     - Shipment dates (`drop_date`, `po_line_edd`) and GST applicability (`gst_applicable_line`) define delivery terms.  
     - Versioning is tracked via `version_timestamp`, `version_editor`, and `version_number`.  
     - Order status is managed with `cancelled` and `received` (non-lot items) flags, while authorization controls (`non_preferred_auth`, `price_increase_auth`, `auth_time_price`) and supplementary notes (`pol_purchasing_notes`, `pol_internal_notes`, `pol_freight_tracking`, `pol_risk`, `pol_inspection_instructions`, `pol_supplier_requirements`, `pol_quote_ref`) support internal controls and risk management.  
     - **Joins:**  
       - `purchase_order_lines.po_id = purchase_orders.id` (each line belongs to a purchase order).  
       - `purchase_order_lines.lot_no = comp_lot.lot_no` (if a line references a specific inventory lot).  

2. **purchase_orders**  
   - **Description:** Stores purchase order details covering vendor info, authorization, shipping, and financial terms.  
   - **COMMENT Section:**  
     - Key fields include `id` (PK), `pono`, `date_issued`, `revision`, `vendor_id`, `currency`, `conversion_rate`, and `freight_value`.  
     - Tracks order revisions, approvals (`auth_id`, `auth_date`, etc.), and associated notes (`amended_notes`, `order_comments`, `delivery_comments`, `notes`).  
     - Includes vendor contact details (`vendor_name`, `attention`, `address`, `vendor_email`, `vendor_fax`), shipping methods (`shipping`, `shipping_extra`), and incoterms.  
     - Contains fields for document attachments and invoicing (`attachments`, `po_email_attachments`, `po_invoicing_check`) plus internal payment terms and status flags (`po_status`, `po_closed`).  
     - Version control is maintained via `version_number`, `version_editor`, and `version_timestamp`.  
     - **Joins:**  
       - `purchase_order_lines.po_id = purchase_orders.id` (each line belongs to a purchase order).  
       - `comp_lot.purchase_order = purchase_orders.po_no` (each lot was purchased on a purchase order).  
       - `purchase_orders.vendor_id = Company.CompanyID`.  

---

#### Key Relationships:
1. **purchase_order_lines → purchase_orders**  
   - Join Condition: `purchase_order_lines.po_id = purchase_orders.id`  
   - Description: Links each line item in `purchase_order_lines` to its parent purchase order in `purchase_orders`.  

2. **purchase_order_lines → comp_lot**  
   - Join Condition: `purchase_order_lines.lot_no = comp_lot.lot_no`  
   - Description: Links a line item to a specific inventory lot, if applicable.  

3. **purchase_orders → Company**  
   - Join Condition: `purchase_orders.vendor_id = Company.CompanyID`  
   - Description: Links a purchase order to its associated vendor in the `Company` table.  

4. **comp_lot → purchase_orders**  
   - Join Condition: `comp_lot.purchase_order = purchase_orders.po_no`  
   - Description: Links inventory lots to the purchase orders they were purchased on.  

---

#### Relevant Fields for the User Question:
1. **From `purchase_orders`:**  
   - `id`: Primary key of the purchase order.  
   - `pono`: Visual purchase order number.  
   - `date_issued`: Date the purchase order was issued.  
   - `revision`: Revision number of the purchase order.  
   - `revision_date`: Date the purchase order was last revised.  
   - `po_status`: Status of the purchase order (e.g., Current or Cancelled).  

2. **From `purchase_order_lines`:**  
   - `po_id`: Foreign key linking to `purchase_orders.id`.  
   - `drop_date`: Expected shipping and invoicing date from the vendor.  
   - `po_line_edd`: Expected delivery date of the part.  
    """

    #insert_data(collection, doc1)
    #insert_data(collection, doc2)
    #insert_data(collection, doc3)

