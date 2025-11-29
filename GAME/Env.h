#ifndef ENV_H
#define ENV_H

#include <stdio.h>
#include <stdlib.h>


class Env {
public:
  std::vector<std::string> env;
  Env(std::vector<std::string> env) {
    this->env = env;
  }
};

#endif
