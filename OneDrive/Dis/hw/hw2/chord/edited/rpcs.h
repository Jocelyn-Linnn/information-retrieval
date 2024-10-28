#ifndef RPCS_H
#define RPCS_H

#include "chord.h"
#include "rpc/client.h"

Node self, successor, predecessor;
const int m = 32; // 假設 m = 32，對應 32 位的 ID 空間
std::vector<Node> finger_table(m); // Finger Table 大小為 m
int next = 0;


Node get_info() { return self; } // Do not modify this line.

// 初始化 Finger Table
void init_finger_table() {
  for (int i = 0; i < m; ++i) {
    finger_table[i] = self;
  }
}

void create() {
  predecessor.ip = "";
  successor = self;
  init_finger_table(); // 初始化 Finger Table
  std::cout << "create success!";
}

// 假如A要joinB，A呼叫B裡的join function。這邊的self是A。
void join(Node n) {
  predecessor.ip = "";
  rpc::client client(n.ip, n.port);
  successor = client.call("find_successor", self.id).as<Node>();
  std::cout << "join success!";
}

// 根據 Finger Table 查找最接近但小於 id 的節點
Node closest_preceding_node(uint64_t id) {
  // 從 Finger Table 的尾端開始查找
  for (int i = m - 1; i >= 0; --i) {
    if (finger_table[i].id > self.id && finger_table[i].id < id) {
      return finger_table[i];
    }
  }
  return self;
}

// B裡的join function被啟動，參數是A的id。這邊的self是B。
Node find_successor(uint64_t id) {
  if (id > self.id && id <= successor.id) {
    return successor;
  } else {
    Node closest_preceding = closest_preceding_node(id);
    rpc::client client(closest_preceding.ip, closest_preceding.port);
    return client.call("find_successor", id).as<Node>();
  }
}

Node get_predecessor() {
  return predecessor;
}

void stabilize() {
  rpc::client client(successor.ip, successor.port);
  Node x = client.call("get_predecessor").as<Node>();

  if (x.ip != "" && (x.id > self.id && x.id < successor.id)) {
    successor = x;
  }

  client.call("notify", self);
}

void notify(Node n) {
  if (predecessor.ip == "" || (n.id > predecessor.id && n.id < self.id)) {
    predecessor = n;
  }
}

void fix_fingers() {
  next = next + 1;
  if (next >= m) {
    next = 0;
  }
  uint64_t start = (self.id + (1ULL << next)) % (1ULL << m);
  finger_table[next] = find_successor(start);
}

void check_predecessor() {
  try {
    rpc::client client(predecessor.ip, predecessor.port);
    Node n = client.call("get_info").as<Node>();
  } catch (std::exception &e) {
    predecessor.ip = "";
  }
}

void register_rpcs() {
  add_rpc("get_info", &get_info); // Do not modify this line.
  add_rpc("create", &create);
  add_rpc("join", &join);
  add_rpc("find_successor", &find_successor);
  add_rpc("notify", &notify);
  add_rpc("get_predecessor", &get_predecessor); 
}

void register_periodics() {
  add_periodic(check_predecessor);
  add_periodic(stabilize);
  add_periodic(fix_fingers); // 定期更新 Finger Table
}

#endif /* RPCS_H */
