#!/usr/bin/python3

import msgpackrpc
import time

def new_client(ip, port):
	return msgpackrpc.Client(msgpackrpc.Address(ip, port))

client_0 = new_client("127.0.0.1", 5056)
client_1 = new_client("127.0.0.1", 5057)
client_2 = new_client("127.0.0.1", 5058)
client_3 = new_client("127.0.0.1", 5059)
client_4 = new_client("127.0.0.1", 5060)
client_5 = new_client("127.0.0.1", 5061)

print(client_0.call("get_info"))
print(client_1.call("get_info"))
print(client_2.call("get_info"))
print(client_3.call("get_info"))
print(client_4.call("get_info"))
print(client_5.call("get_info"))

client_0.call("create")

client_2.call("join", client_0.call("get_info"))
client_1.call("join", client_2.call("get_info"))
client_4.call("join", client_1.call("get_info"))
client_5.call("join", client_0.call("get_info"))
client_3.call("join", client_4.call("get_info"))

# test the functionality after all nodes have joined the Chord ring
ans = client_0.call("find_successor", 1234567)
ans = client_1.call("find_successor", 1234567890)
