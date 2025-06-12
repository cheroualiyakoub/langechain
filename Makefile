# Makefile for PatentMuse
.PHONY: start up down logs urls status clean help check-services

# Variables
COMPOSE_FILE = docker-compose.yaml
HOST = localhost

# Colors for display
GREEN = \033[0;32m
RED = \033[0;31m
BLUE = \033[0;34m
YELLOW = \033[1;33m
CYAN = \033[0;36m
NC = \033[0m # No Color

# Main command - Launches docker-compose and displays URLs with status check
start:
	@echo "$(CYAN)🚀 Starting PatentMuse...$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	docker-compose -f $(COMPOSE_FILE) up -d
	@echo "$(YELLOW)⏳ Waiting for services to start...$(NC)"
	@sleep 8
	@make check-services
	@make urls

# Service status check with colored output
check-services:
	@echo "\n$(CYAN)🔍 Checking service status...$(NC)"
	@echo "$(BLUE)┌─────────────────────────────────────────────────────────┐$(NC)"
	@if curl -s http://$(HOST):8000/health > /dev/null 2>&1; then \
		echo "$(BLUE)│$(NC) $(GREEN)✅ Langchain API$(NC)     $(GREEN)[ONLINE]$(NC)  Port 8000    $(BLUE)│$(NC)"; \
	else \
		echo "$(BLUE)│$(NC) $(RED)❌ Langchain API$(NC)     $(RED)[OFFLINE]$(NC) Port 8000    $(BLUE)│$(NC)"; \
	fi
	@if curl -s http://$(HOST):8001 > /dev/null 2>&1; then \
		echo "$(BLUE)│$(NC) $(GREEN)✅ Chroma Vector DB$(NC)  $(GREEN)[ONLINE]$(NC)  Port 8001    $(BLUE)│$(NC)"; \
	else \
		echo "$(BLUE)│$(NC) $(RED)❌ Chroma Vector DB$(NC)  $(RED)[OFFLINE]$(NC) Port 8001    $(BLUE)│$(NC)"; \
	fi
	@if curl -s http://$(HOST):8888 > /dev/null 2>&1; then \
		echo "$(BLUE)│$(NC) $(GREEN)✅ Jupyter Lab$(NC)       $(GREEN)[ONLINE]$(NC)  Port 8888    $(BLUE)│$(NC)"; \
	else \
		echo "$(BLUE)│$(NC) $(RED)❌ Jupyter Lab$(NC)       $(RED)[OFFLINE]$(NC) Port 8888    $(BLUE)│$(NC)"; \
	fi
	@echo "$(BLUE)└─────────────────────────────────────────────────────────┘$(NC)"

# Display all service URLs with improved style
urls:
	@echo "\n$(GREEN)🌐 PatentMuse service URLs$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(CYAN)┌─────────────────────────────────────────────────────────┐$(NC)"
	@echo "$(CYAN)│$(NC) $(GREEN)🔗 Langchain API:$(NC)     http://$(HOST):8000         $(CYAN)│$(NC)"
	@echo "$(CYAN)│$(NC) $(GREEN)📊 Chroma Vector DB:$(NC)  http://$(HOST):8001         $(CYAN)│$(NC)"
	@echo "$(CYAN)│$(NC) $(GREEN)📓 Jupyter Lab:$(NC)       http://$(HOST):8888         $(CYAN)│$(NC)"
	@echo "$(CYAN)└─────────────────────────────────────────────────────────┘$(NC)"
	@echo "\n$(YELLOW)🎯 Useful API endpoints:$(NC)"
	@echo "  $(GREEN)•$(NC) Health check:    http://$(HOST):8000/health"
	@echo "  $(GREEN)•$(NC) Chat endpoint:   http://$(HOST):8000/chat"
	@echo "  $(GREEN)•$(NC) Providers:       http://$(HOST):8000/providers"
	@echo "  $(GREEN)•$(NC) Chroma docs:     http://$(HOST):8001/docs"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)✨ PatentMuse is ready! Happy coding! ✨$(NC)\n"

# Alias for start command (backward compatibility)
up: start

# Stop all services with confirmation
down:
	@echo "$(RED)🛑 Stopping PatentMuse...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down
	@echo "$(YELLOW)✅ All services have been stopped$(NC)"

# Display real-time logs with colors
logs:
	@echo "$(CYAN)📋 Real-time logs...$(NC)"
	docker-compose -f $(COMPOSE_FILE) logs -f

# Display container status with style
status:
	@echo "$(GREEN)📊 PatentMuse container status$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	docker-compose -f $(COMPOSE_FILE) ps

# Complete image rebuild
rebuild:
	@echo "$(YELLOW)🔄 Complete PatentMuse rebuild...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down
	docker-compose -f $(COMPOSE_FILE) build --no-cache
	@echo "$(GREEN)✅ Images rebuilt$(NC)"
	@make start

# Complete cleanup
clean:
	@echo "$(RED)🧹 Complete PatentMuse cleanup...$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	
	# Stop Docker containers (keep volumes)
	@echo "$(CYAN)🐳 Stopping Docker containers...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down --remove-orphans 2>/dev/null || true
	
	# Clean orphaned Docker images only
	@echo "$(CYAN)🗑️  Cleaning orphaned Docker images...$(NC)"
	docker image prune -f 2>/dev/null || true
	
	# Remove compiled Python files
	@echo "$(CYAN)🐍 Python cleanup...$(NC)"
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	
	# Remove temporary files
	@echo "$(CYAN)🗑️  Cleaning temporary files...$(NC)"
	find . -type f -name "*.tmp" -delete 2>/dev/null || true
	find . -type f -name "*.temp" -delete 2>/dev/null || true
	find . -type f -name "*.swp" -delete 2>/dev/null || true
	find . -type f -name "*.swo" -delete 2>/dev/null || true
	find . -type f -name "*~" -delete 2>/dev/null || true
	
	# Remove macOS files
	@echo "$(CYAN)🍎 macOS cleanup...$(NC)"
	find . -name ".DS_Store" -delete 2>/dev/null || true
	find . -name ".DS_Store?" -delete 2>/dev/null || true
	find . -name "._*" -delete 2>/dev/null || true
	
	# Remove logs
	@echo "$(CYAN)📋 Log cleanup...$(NC)"
	find . -type f -name "*.log" -delete 2>/dev/null || true
	rm -rf logs/ 2>/dev/null || true
	
	# Remove Jupyter checkpoints
	@echo "$(CYAN)📓 Jupyter cleanup...$(NC)"
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	
	# Remove Python caches
	@echo "$(CYAN)💾 Cache cleanup...$(NC)"
	rm -rf .pytest_cache/ 2>/dev/null || true
	rm -rf .mypy_cache/ 2>/dev/null || true
	rm -rf .ruff_cache/ 2>/dev/null || true
	rm -rf .cache/ 2>/dev/null || true
	
	# Remove backup files
	@echo "$(CYAN)💽 Backup cleanup...$(NC)"
	find . -name "*.backup" -delete 2>/dev/null || true
	find . -name "*.bak" -delete 2>/dev/null || true
	find . -name "*.orig" -delete 2>/dev/null || true
	
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)✅ Cleanup completed! Docker volumes preserved$(NC)"

# Complete cleanup with volume removal (DANGER)
clean-all:
	@echo "$(RED)⚠️  WARNING: Complete cleanup with volume removal!$(NC)"
	@echo "$(RED)This will permanently delete your Chroma data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(RED)🧹 Complete cleanup with volumes...$(NC)"; \
		docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans 2>/dev/null || true; \
		docker system prune -af 2>/dev/null || true; \
		make clean; \
		echo "$(GREEN)✅ Complete cleanup finished$(NC)"; \
	else \
		echo "$(YELLOW)❌ Cleanup cancelled$(NC)"; \
	fi

# Light cleanup (keep Docker)
clean-light:
	@echo "$(YELLOW)🧽 Light PatentMuse cleanup...$(NC)"
	
	# Remove compiled Python files
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	
	# Remove temporary files
	find . -type f -name "*.tmp" -delete 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	
	@echo "$(GREEN)✅ Light cleanup completed$(NC)"

# Restart services
restart:
	@echo "$(YELLOW)🔄 Restarting PatentMuse...$(NC)"
	@make down
	@sleep 2
	@make start

# Quick service test
test:
	@echo "$(CYAN)🧪 Testing PatentMuse services...$(NC)"
	@make check-services

# Display help with style
help:
	@echo "$(GREEN)📖 Available PatentMuse commands$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "  $(GREEN)make start$(NC)      - $(CYAN)Launch all services and display URLs$(NC)"
	@echo "  $(GREEN)make down$(NC)       - $(CYAN)Stop all services$(NC)"
	@echo "  $(GREEN)make urls$(NC)       - $(CYAN)Display service URLs$(NC)"
	@echo "  $(GREEN)make test$(NC)       - $(CYAN)Check service status$(NC)"
	@echo "  $(GREEN)make logs$(NC)       - $(CYAN)Display real-time logs$(NC)"
	@echo "  $(GREEN)make status$(NC)     - $(CYAN)Display container status$(NC)"
	@echo "  $(GREEN)make rebuild$(NC)    - $(CYAN)Rebuild and restart services$(NC)"
	@echo "  $(GREEN)make restart$(NC)    - $(CYAN)Restart all services$(NC)"
	@echo "  $(GREEN)make clean$(NC)      - $(CYAN)Safe cleanup (preserve volumes)$(NC)"
	@echo "  $(GREEN)make clean-all$(NC)  - $(CYAN)Complete cleanup + volumes (DANGER)$(NC)"
	@echo "  $(GREEN)make clean-light$(NC) - $(CYAN)Light cleanup (temporary files)$(NC)"
	@echo "  $(GREEN)make help$(NC)       - $(CYAN)Display this help$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

# Default command
.DEFAULT_GOAL := start