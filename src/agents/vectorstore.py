from pymilvus import DataType, MilvusClient, connections, Collection
from agents.llm import embeddings

DATABASE_URI = "./sql_agent.db"
COLLECTION_NAME = "sql_agent"



def create_collection():
    """
    Create a new collection in Milvus with the specified schema and index parameters.
    This function checks if the collection already exists, and if not, it creates a new one. 
    """
    client = MilvusClient(
        uri=DATABASE_URI,
    )

    if client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return
    
    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4000, enable_analyzer=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1536)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="vector",
        index_name="vector_index",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128}
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    ) 


def get_collection() -> Collection:
    """
    Connect to the Milvus server and get the collection.
    Returns:
        Collection: The Milvus collection.
    """
    connections.connect(
        uri=DATABASE_URI,
        alias="default",
    ) 
    return Collection(COLLECTION_NAME)


def insert_data(collection: Collection, information: str):
    """
    Insert data into the Milvus collection.
    Args:
        collection (Collection): The Milvus collection.
        information (str): The information to insert.
    """
    data = [
        {
            "text": information,
            "vector": embeddings.embed_query(information),
        }
    ]
    collection.insert(data)
    print(f"Inserted data into collection '{COLLECTION_NAME}'.")

def search_data(collection: Collection, query: str, top_k: int = 1):
    """
    Search for data in the Milvus collection.
    Args:
        collection (Collection): The Milvus collection.
        query (str): The query to search for.
        top_k (int): The number of top results to return.
    Returns:
        str: The text of the most relevant result.
    """
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10},
    }
    query_vector = embeddings.embed_query(query)
    result = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["text"],
        expr=None,
    )[0]
    return result[0].entity.get("text") if result else None


if __name__ == "__main__":
    # Get the collection
    create_collection()
    collection = get_collection()
    test = """

    ### Structured Output

    #### User Question:
    Can you provide a list of the top 20 component categories purchased in the last year based on total spend from `comp_lot`? For each category, please include the following details: category name, total spend for the category, the ID with the highest spend in that category, and the total spend for that ID.

    ---

    ### Tables and Schemas:

    #### Table: `comp_lot`
    - **Table Description**: Tracks detailed information for component lots, providing batch-level visibility into inventory and sourcing. Key fields include `lot_no` (human-readable identifier), `lot_date` (creation date), `price` (unit price in AUD), `purchased_amount` (quantity purchased), and `source_ID` (link to sourcing record). This table supports inventory and sourcing traceability.
    
    - **Schema**:
    sql

    CREATE TABLE comp_lot (
        lot_no VARCHAR(20) NOT NULL COMMENT 'Key field' DEFAULT '', 
        lot_date DATE COMMENT 'Date lot was created. NOT when it was purchased.', 
        price FLOAT COMMENT 'Unit price, converted to AUD', 
        purchased_amount FLOAT COMMENT 'How many were purchased. If the lot is split, this number reduces by the split amount.' DEFAULT 0, 
        source_ID INTEGER(11) COMMENT 'JOIN comp_lot.source_ID = comp_src.source_ID' DEFAULT 0, 
        PRIMARY KEY (lot_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE utf8mb3_general_ci;

    - **COMMENT Section**:
    Tracks detailed information for component lots, providing batch-level visibility into inventory and sourcing. The primary key `lot_id` uniquely identifies each lot, while `lot_no` serves as a human-readable identifier formatted as `DDMMYY-X`, with optional suffixes for split lots. Key fields include `lot_date` (creation date), `price` (per part in AUD), and `purchased_amount` (acquisition details). The `source_ID` field links each lot to its sourcing record in `comp_src`, ensuring traceability.

    **Joins**:
    - `comp_src.source_ID = comp_lot.source_ID` (each sourcing record links to a specific lot via `source_ID`).
    - `purchase_order_lines.lot_no = comp_lot.lot_no` (if a line references a specific inventory lot).
    - `comp_lot.purchase_order = purchase_orders.pono` (each lot was purchased on a particular PO).

    - **Relevant Fields**:
    - `lot_no`
    - `lot_date`
    - `price`
    - `purchased_amount`
    - `source_ID`

    ---

    #### Table: `purchase_order_lines`
    - **Table Description**: Tracks individual line items for purchase orders, providing detailed insight into each order’s components and pricing. Key fields include `po_id` (link to parent order), `lot_no` (optional link to inventory lot), `price` (price in PO currency), and `qty` (quantity ordered).

    - **Schema**:
    sql

    CREATE TABLE purchase_order_lines (
        po_id INTEGER(11) COMMENT 'JOIN purchase_order_lines.po_id = purchase_orders.id', 
        lot_no VARCHAR(20) COMMENT 'JOIN comp_lot.lot_no = purchase_order_lines.lot_no', 
        price DECIMAL(15, 5) COMMENT 'Price in the PO currency.', 
        qty FLOAT COMMENT 'Quantity ordered', 
        PRIMARY KEY (id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE utf8mb3_general_ci;

    - **COMMENT Section**:
    Tracks individual line items for purchase orders, providing detailed insight into each order’s components and pricing. The primary key `id` uniquely identifies each line, and `po_id` links the line to its parent order, while `lot_no` optionally ties the line to a specific inventory lot. Key fields include `price` (price in PO currency) and `qty` (quantity ordered).

    **Joins**:
    - `purchase_order_lines.po_id = purchase_orders.id` (each line belongs to a purchase order).
    - `purchase_order_lines.lot_no = comp_lot.lot_no` (if a line references a specific inventory lot).

    - **Relevant Fields**:
    - `po_id`
    - `lot_no`
    - `price`
    - `qty`

    ---

    #### Table: `category`
    - **Table Description**: Categorizes components into different categories (e.g., integrated circuit, capacitor, resistor, metalwork). Key fields include `ID` (primary key) and `category` (text field indicating the component category).

    - **Schema**:
    sql

    CREATE TABLE category (
        ID INTEGER(11) NOT NULL COMMENT 'JOIN category.ID = component.cat_no' AUTO_INCREMENT, 
        category VARCHAR(100) COMMENT 'Text field showing the component category' DEFAULT '', 
        PRIMARY KEY (ID)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE utf8mb3_general_ci;

    - **COMMENT Section**:
    The `category` table is used to categorize components into different categories (e.g., integrated circuit, capacitor, resistor, metalwork). The key field is `ID`, and the `category` field is a text field indicating the component category.

    **Joins**:
    - `component.cat_no = category.ID` (each component is of a particular category).

    - **Relevant Fields**:
    - `ID`
    - `category`

    ---

    #### Table: `component`
    - **Table Description**: Stores component data representing groups of parts conforming to specific specifications. Key fields include `ID` (primary key) and `cat_no` (link to `category.ID`).

    - **Schema**:
    sql

    CREATE TABLE component (
        ID INTEGER(11) NOT NULL COMMENT 'Primary key and text identifier of the component. e.g. ID6334' AUTO_INCREMENT, 
        cat_no INTEGER(11) COMMENT 'JOIN component.cat_no = category.ID' DEFAULT 0, 
        PRIMARY KEY (ID)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE utf8mb3_general_ci;

    - **COMMENT Section**:
    Stores component data representing groups of parts conforming to specific specifications. Each component is uniquely identified (e.g., `ID6334`) and includes a descriptive field detailing its purpose and application. Components are categorized (e.g., capacitor, resistor, end product, subassembly) via the `component.cat_no` field, which links to `category.ID` for added context.

    **Joins**:
    - `component.cat_no = category.ID` (each component is of a particular category).
    - `component.ID = comp_src.ID` (each sourcing record is linked to a component via its ID).

    - **Relevant Fields**:
    - `ID`
    - `cat_no`

    ---

    #### Table: `comp_src`
    - **Table Description**: Stores sourcing details for components, linking each component’s ID to its associated vendor and manufacturer data. Key fields include `source_ID` (primary key) and `ID` (link to `component.ID`).

    - **Schema**:
    sql

    CREATE TABLE comp_src (
        source_ID INTEGER(11) NOT NULL COMMENT 'primary key for the comp_src table' AUTO_INCREMENT, 
        ID INTEGER(11) COMMENT 'JOIN comp_src.ID = component.ID', 
        PRIMARY KEY (source_ID)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE utf8mb3_general_ci;

    - **COMMENT Section**:
    Stores sourcing details for components, linking each component’s ID to its associated vendor and manufacturer data. The `source_ID` field connects sourcing records to specific component lots in `comp_lot`, enhancing traceability.

    **Joins**:
    - `comp_src.ID = component.ID` (each sourcing record is linked to a component via its ID).
    - `comp_src.source_ID = comp_lot.source_ID` (each sourcing record links to a specific lot via `source_ID`).

    - **Relevant Fields**:
    - `source_ID`
    - `ID`

    ---

    ### Key Relationships:
    1. **`comp_lot` ↔ `comp_src`**:
    - `comp_lot.source_ID = comp_src.source_ID` (links lots to sourcing records).

    2. **`comp_src` ↔ `component`**:
    - `comp_src.ID = component.ID` (links sourcing records to components).

    3. **`component` ↔ `category`**:
    - `component.cat_no = category.ID` (links components to their categories).

    4. **`comp_lot` ↔ `purchase_order_lines`**:
    - `comp_lot.lot_no = purchase_order_lines.lot_no` (links lots to purchase order lines).

    ---

    ### Relevant Fields:
    - **From `comp_lot`**: `lot_no`, `lot_date`, `price`, `purchased_amount`, `source_ID`
    - **From `purchase_order_lines`**: `po_id`, `lot_no`, `price`, `qty`
    - **From `category`**: `ID`, `category`
    - **From `component`**: `ID`, `cat_no`
    - **From `comp_src`**: `source_ID`, `ID`
    """
    # insert
    insert_data(collection, test)
    result = search_data(collection, "comp lot")
    print(f"Search result: {result}")
