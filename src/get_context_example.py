from agents.vectorstore import get_collection

def main():
    # Example usage of get_collection
    collection = get_collection("intellidesign.db", "sql_agent")
    result = collection.query(expr="id >= 0", output_fields=["text"])
    print(len(result))
if __name__ == "__main__":
    main()