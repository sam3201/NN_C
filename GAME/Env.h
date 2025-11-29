#ifndef ENV_H
#define ENV_H

#include <vector>
#include <string>

class Env {
public:
  std::vector<std::string> env;
  Env(std::vector<std::string> env) {
    this->env = env;
  }
};

#endif
