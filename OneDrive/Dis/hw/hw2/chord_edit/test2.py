#!/usr/bin/python3

import time
import msgpackrpc

# 創建新的 RPC 客戶端
def new_client(ip, port):
    return msgpackrpc.Client(msgpackrpc.Address(ip, port))

# 初始化節點客戶端
client_1 = new_client("127.0.0.1", 5057)
client_2 = new_client("127.0.0.1", 5058)
client_3 = new_client("127.0.0.1", 5059)

# Step 1: create RPC
print("Calling 'create' on client_1")
client_1.call("create")
print(client_1.call("get_info"))
print("Waiting 2 seconds after 'create'")
time.sleep(2)

# Step 2: join RPC
print("Calling 'join' on client_2 to join client_1")
client_2.call("join", client_1.call("get_info"))
print(client_2.call("get_info"))
print("Waiting 2 seconds after 'join'")
time.sleep(2)

print("Calling 'join' on client_3 to join client_1")
client_3.call("join", client_1.call("get_info"))
print(client_3.call("get_info"))
print("Waiting 2 seconds after 'join'")
time.sleep(2)

# Step 3: Testing 'find_successor' after 20 seconds
print("Waiting 20 seconds before calling 'find_successor'")
time.sleep(20)
print("Calling 'find_successor' on client_2 with key 123")
successor_1 = client_2.call("find_successor", 123)
print(f"Successor of key 123 from client_2: {successor_1}")

print("Calling 'find_successor' on client_3 with key 456")
successor_2 = client_3.call("find_successor", 456)
print(f"Successor of key 456 from client_3: {successor_2}")

# Step 4: 再次調用 'get_info'
print("Calling 'get_info' on client_1")
info_1 = client_1.call("get_info")
print(info_1)

print("Calling 'get_info' on client_2")
info_2 = client_2.call("get_info")
print(info_2)

print("Calling 'get_info' on client_3")
info_3 = client_3.call("get_info")
print(info_3)
