import sys
import psycopg2

print("1. A iniciar o teste de conexao...", flush=True)

try:
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="secret",
        host="127.0.0.1",  # Usar 127.0.0.1 diretamente em vez de localhost
        port="5432"
    )
    print("2. ✅ CONECTOU COM SUCESSO NO POSTGRESQL!", flush=True)
    
    cur = conn.cursor()
    cur.execute("SELECT version();")
    row = cur.fetchone()
    print("3. Versao do banco:", row[0], flush=True)
    
    conn.close()
    print("4. Conexao fechada com sucesso.", flush=True)

except Exception as err:
    print("❌ Falha na conexao:", flush=True)
    print(type(err).__name__, repr(err), flush=True)