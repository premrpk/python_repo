import sqlglot
from sqlglot import exp
from collections import defaultdict

def extract_table_info(sql):
    """Extract table information from SQL using sqlglot"""
    try:
        # Parse the SQL statement
        parsed = sqlglot.parse_one(sql, read=None)
        
        # Structure to store table info: {table_name: {'alias': alias, 'cte': True/False}}
        table_info = defaultdict(dict)
        
        # Extract CTEs first
        ctes = set()
        for cte in parsed.find_all(exp.CTE):
            ctes.add(cte.alias_or_name)
        
        # Find all table references
        for table in parsed.find_all(exp.Table):
            table_name = table.name
            alias = table.alias_or_name
            
            # Skip if this is a reference to a CTE
            if table_name in ctes:
                continue
                
            # Store table info
            if table_name not in table_info:
                table_info[table_name]['alias'] = None
                table_info[table_name]['cte'] = False
            
            # Update alias if present
            if alias and alias != table_name:
                table_info[table_name]['alias'] = alias
        
        # Find all subqueries and process them recursively
        for subquery in parsed.find_all(exp.Subquery):
            subquery_tables = extract_table_info(subquery.sql())
            for table_name, info in subquery_tables.items():
                if table_name not in table_info:
                    table_info[table_name] = info
        
        return dict(table_info)
    
    except sqlglot.errors.ParseError as e:
        print(f"Error parsing SQL: {e}")
        return {}

def process_sql_file(file_path):
    """Process SQL file and extract table information"""
    with open(file_path, 'r') as f:
        sql_content = f.read()
    
    # Split into individual statements if needed
    statements = sqlglot.parse(sql_content, read=None)
    
    all_tables = defaultdict(dict)
    
    for stmt in statements:
        if not isinstance(stmt, exp.Select):
            continue
            
        tables = extract_table_info(stmt.sql())
        for table_name, info in tables.items():
            if table_name not in all_tables:
                all_tables[table_name] = info
    
    return dict(all_tables)

def print_table_info(table_info):
    """Print table information in a readable format"""
    print("\nTable References:")
    print("{:<30} {:<20} {:<10}".format("Table Name", "Alias", "Is CTE"))
    print("-" * 60)
    for table_name, info in sorted(table_info.items()):
        print("{:<30} {:<20} {:<10}".format(
            table_name,
            info.get('alias', ''),
            str(info.get('cte', False))
        ))

if __name__ == "__main__":
    file_path = input("Enter path to SQL file: ")
    try:
        table_info = process_sql_file(file_path)
        print_table_info(table_info)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
