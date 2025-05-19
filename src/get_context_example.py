from agents.vectorstore import get_collection
import json

def main():
    # Example usage of get_collection
    collection = get_collection("intellidesign.db", "sql_agent")
    result = collection.query(expr="id >= 0", output_fields=["text"])
    # write to json file
    with open("output.json", "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()