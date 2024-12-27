.PHONY: all install clean test

all: install

install:
	@echo "Installing cropcircles library..."
	pip install -e .

# Clean up build artifacts
clean:
	@echo "Cleaning up build artifacts..."
	rm -rf build/ dist/ crop-circles.egg-info/

# Run tests (not present by now)
test:
	@echo "Running tests..."
	pytest tests/ 

# Show help
help:
	@echo "Makefile for the cropcircles library"
	@echo "Available targets:"
	@echo "  all       - Install the library"
	@echo "  install   - Install the library in editable mode"
	@echo "  clean     - Remove build artifacts"
	@echo "  test      - Run tests"
	@echo "  help      - Show this help message"

# After installation we can import the library in python
# import cropcircles as cc