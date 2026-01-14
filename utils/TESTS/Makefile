# Makefile for NN Framework Tests
# Usage: make all, make test, make clean

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -I../NN
LDFLAGS = -lm

# Source files (excluding NN.c to avoid duplicate symbols)
NN_SOURCES = ../NN/NN/NN.c ../NN/CONVOLUTION/CONVOLUTION.c ../NN/TRANSFORMER/TRANSFORMER.c ../NN/NEAT/NEAT.c ../NN/TOKENIZER/TOKENIZER.c

# Test executables
TESTS = test_nn_enhanced test_convolution_integration test_transformer_integration test_neat_integration test_rnn_lstm test_advanced_networks test_muz_enhanced
TEST_EXECUTABLES = $(addsuffix .test,$(TESTS))

# Default target
all: $(TEST_EXECUTABLES)

# Build individual tests
test_nn_enhanced.test: test_nn_enhanced.c
	$(CC) $(CFLAGS) -o $@ $< $(NN_SOURCES) $(LDFLAGS)

test_convolution_integration.test: test_convolution_integration.c
	$(CC) $(CFLAGS) -o $@ $< $(NN_SOURCES) $(LDFLAGS)

test_transformer_integration.test: test_transformer_integration.c
	$(CC) $(CFLAGS) -o $@ $< $(NN_SOURCES) $(LDFLAGS)

test_neat_integration.test: test_neat_integration.c
	$(CC) $(CFLAGS) -o $@ $< $(NN_SOURCES) $(LDFLAGS)

test_rnn_lstm.test: test_rnn_lstm.c
	$(CC) $(CFLAGS) -o $@ $< ../NN/NN.c ../NN/RNN/RNN.c $(LDFLAGS)

test_advanced_networks.test: test_advanced_networks_basic.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_muz_enhanced.test: test_muz_enhanced.c
	$(CC) $(CFLAGS) -o $@ $< ../NN/NN/NN.c ../NN/MUZE/discrete_actions.c ../NN/MUZE/compressed_obs.c ../NN/MUZE/muze_enhanced_config.c ../NN/MUZE/muze_enhanced_model.c $(LDFLAGS)

# Test runner
run_all_tests.test: run_all_tests.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Run all tests
test: all
	@echo "=== Running All NN Framework Tests ==="
	@for test in $(TEST_EXECUTABLES); do \
		echo "Running $$test..."; \
		./$$test; \
		echo ""; \
	done
	@echo "=== Test Summary ==="
	@echo "All tests completed!"

# Run individual tests
test-enhanced: test_nn_enhanced.test
	./test_nn_enhanced.test

test-convolution: test_convolution_integration.test
	./test_convolution_integration.test

test-transformer: test_transformer_integration.test
	./test_transformer_integration.test

test-neat: test_neat_integration.test
	./test_neat_integration.test

test-rnn: test_rnn_lstm.test
	./test_rnn_lstm.test

test-advanced: test_advanced_networks.test
	./test_advanced_networks.test

test-muz: test_muz_enhanced.test
	./test_muz_enhanced.test

# Build and run test runner
test-runner: run_all_tests.test
	./run_all_tests.test

# Clean build artifacts
clean:
	rm -f *.test
	rm -f run_all_tests.test

# Install dependencies (placeholder)
install-deps:
	@echo "No external dependencies required for tests"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all tests"
	@echo "  test         - Run all tests"
	@echo "  test-enhanced - Run enhanced NN tests"
	@echo "  test-convolution - Run CONVOLUTION tests"
	@echo "  test-transformer - Run TRANSFORMER tests"
	@echo "  test-neat    - Run NEAT tests"
	@echo "  test-rnn     - Run RNN/LSTM tests"
	@echo "  test-advanced - Run advanced networks tests"
	@echo "  test-muz - Run enhanced MUZE tests"
	@echo "  test-runner  - Run test runner"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help"

# Phony targets
.PHONY: all test test-enhanced test-convolution test-transformer test-neat test-rnn test-advanced test-muz test-runner clean install-deps help
