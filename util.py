
"""
Pretty print a dictionary
"""
def print_dict(dict_x, depth=0):
    for key, value in dict_x.items():
        if type(value) is dict:
            print_dict(value, depth + 1)
        elif type(value) is list:
            for i in value:
                print("[" + str(i) + "]")
        else:
            print("   " * depth, key + ":")
    print("-----------------------------------")
    for key, value in dict_x.items():
        if type(value) is dict:
            print_dict(value, depth + 1)
        elif type(value) is list:
            print("   " * depth, key + ":")
            for i in value:
                print("   " * depth, "[" + str(i) + "]")
                print("$$$")
        else:
            print("   " * depth, key + ":", value)
        print()
"""
Make sure a name is suitable for a file name
"""
## 
def valid_fname(fname):
    return "".join(x for x in fname if x.isalnum() or x in "_- ")

"""
Dump a sqlite3 database 
"""
import sqlite3
def dump_sqlite3(path = 'example1/.chroma'):
    conn = sqlite3.connect(path +'/' + 'chroma.sqlite3')
    c = conn.cursor()
    # c.execute("SELECT * FROM documents")
    # c.execute("SELECT name FROM sqlite_schema where type='table' ORDER BY name")
    c.execute("PRAGMA table_info(embedding_fulltext_search_data)")
    # c.execute("SELECT * FROM collections")
    c.execute("SELECT * FROM embedding_fulltext_search_data")
    rows = c.fetchall()
    print("count: ", len(rows))
    for row in rows:
        print(row[0], len(row[1]))
    conn.close()

# if __name__ == "__main__":
#   main()