# Makefile pour PatentMuse
.PHONY: start up down logs urls status clean help check-services

# Variables
COMPOSE_FILE = docker-compose.yaml
HOST = localhost

# Couleurs pour l'affichage
GREEN = \033[0;32m
RED = \033[0;31m
BLUE = \033[0;34m
YELLOW = \033[1;33m
CYAN = \033[0;36m
NC = \033[0m # No Color

# Commande principale - Lance docker-compose et affiche les URLs avec vérification de statut
start:
	@echo "$(CYAN)🚀 Démarrage de PatentMuse...$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	docker-compose -f $(COMPOSE_FILE) up -d
	@echo "$(YELLOW)⏳ Attente du démarrage des services...$(NC)"
	@sleep 8
	@make check-services
	@make urls

# Vérification du statut des services avec affichage coloré
check-services:
	@echo "\n$(CYAN)🔍 Vérification du statut des services...$(NC)"
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

# Affiche les URLs de tous les services avec style amélioré
urls:
	@echo "\n$(GREEN)🌐 URLs des services PatentMuse$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(CYAN)┌─────────────────────────────────────────────────────────┐$(NC)"
	@echo "$(CYAN)│$(NC) $(GREEN)🔗 Langchain API:$(NC)     http://$(HOST):8000         $(CYAN)│$(NC)"
	@echo "$(CYAN)│$(NC) $(GREEN)📊 Chroma Vector DB:$(NC)  http://$(HOST):8001         $(CYAN)│$(NC)"
	@echo "$(CYAN)│$(NC) $(GREEN)📓 Jupyter Lab:$(NC)       http://$(HOST):8888         $(CYAN)│$(NC)"
	@echo "$(CYAN)└─────────────────────────────────────────────────────────┘$(NC)"
	@echo "\n$(YELLOW)🎯 Endpoints API utiles :$(NC)"
	@echo "  $(GREEN)•$(NC) Health check:    http://$(HOST):8000/health"
	@echo "  $(GREEN)•$(NC) Chat endpoint:   http://$(HOST):8000/chat"
	@echo "  $(GREEN)•$(NC) Providers:       http://$(HOST):8000/providers"
	@echo "  $(GREEN)•$(NC) Chroma docs:     http://$(HOST):8001/docs"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)✨ PatentMuse est prêt ! Bon travail ! ✨$(NC)\n"

# Alias pour la commande start (rétrocompatibilité)
up: start

# Arrête tous les services avec confirmation
down:
	@echo "$(RED)🛑 Arrêt de PatentMuse...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down
	@echo "$(YELLOW)✅ Tous les services ont été arrêtés$(NC)"

# Affiche les logs en temps réel avec couleurs
logs:
	@echo "$(CYAN)📋 Logs en temps réel...$(NC)"
	docker-compose -f $(COMPOSE_FILE) logs -f

# Affiche le statut des conteneurs avec style
status:
	@echo "$(GREEN)📊 Statut des conteneurs PatentMuse$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	docker-compose -f $(COMPOSE_FILE) ps

# Reconstruction complète des images
rebuild:
	@echo "$(YELLOW)🔄 Reconstruction complète de PatentMuse...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down
	docker-compose -f $(COMPOSE_FILE) build --no-cache
	@echo "$(GREEN)✅ Images reconstruites$(NC)"
	@make start

# Nettoyage complet
clean:
	@echo "$(RED)🧹 Nettoyage complet de PatentMuse...$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	
	# Arrêt des conteneurs Docker (garde les volumes)
	@echo "$(CYAN)🐳 Arrêt des conteneurs Docker...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down --remove-orphans 2>/dev/null || true
	
	# Nettoyage des images Docker orphelines uniquement
	@echo "$(CYAN)🗑️  Nettoyage images Docker orphelines...$(NC)"
	docker image prune -f 2>/dev/null || true
	
	# Suppression des fichiers Python compilés
	@echo "$(CYAN)🐍 Nettoyage Python...$(NC)"
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	
	# Suppression des fichiers temporaires
	@echo "$(CYAN)🗑️  Nettoyage fichiers temporaires...$(NC)"
	find . -type f -name "*.tmp" -delete 2>/dev/null || true
	find . -type f -name "*.temp" -delete 2>/dev/null || true
	find . -type f -name "*.swp" -delete 2>/dev/null || true
	find . -type f -name "*.swo" -delete 2>/dev/null || true
	find . -type f -name "*~" -delete 2>/dev/null || true
	
	# Suppression des fichiers macOS
	@echo "$(CYAN)🍎 Nettoyage macOS...$(NC)"
	find . -name ".DS_Store" -delete 2>/dev/null || true
	find . -name ".DS_Store?" -delete 2>/dev/null || true
	find . -name "._*" -delete 2>/dev/null || true
	
	# Suppression des logs
	@echo "$(CYAN)📋 Nettoyage logs...$(NC)"
	find . -type f -name "*.log" -delete 2>/dev/null || true
	rm -rf logs/ 2>/dev/null || true
	
	# Suppression des checkpoints Jupyter
	@echo "$(CYAN)📓 Nettoyage Jupyter...$(NC)"
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	
	# Suppression des caches Python
	@echo "$(CYAN)💾 Nettoyage caches...$(NC)"
	rm -rf .pytest_cache/ 2>/dev/null || true
	rm -rf .mypy_cache/ 2>/dev/null || true
	rm -rf .ruff_cache/ 2>/dev/null || true
	rm -rf .cache/ 2>/dev/null || true
	
	# Suppression des fichiers de sauvegarde
	@echo "$(CYAN)💽 Nettoyage sauvegardes...$(NC)"
	find . -name "*.backup" -delete 2>/dev/null || true
	find . -name "*.bak" -delete 2>/dev/null || true
	find . -name "*.orig" -delete 2>/dev/null || true
	
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)✅ Nettoyage terminé ! Volumes Docker préservés$(NC)"

# Nettoyage complet avec suppression des volumes (DANGER)
clean-all:
	@echo "$(RED)⚠️  ATTENTION: Nettoyage complet avec suppression des volumes!$(NC)"
	@echo "$(RED)Ceci supprimera définitivement vos données Chroma!$(NC)"
	@read -p "Êtes-vous sûr? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(RED)🧹 Nettoyage complet avec volumes...$(NC)"; \
		docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans 2>/dev/null || true; \
		docker system prune -af 2>/dev/null || true; \
		make clean; \
		echo "$(GREEN)✅ Nettoyage complet terminé$(NC)"; \
	else \
		echo "$(YELLOW)❌ Nettoyage annulé$(NC)"; \
	fi

# Nettoyage léger (garde Docker)
clean-light:
	@echo "$(YELLOW)🧽 Nettoyage léger de PatentMuse...$(NC)"
	
	# Suppression des fichiers Python compilés
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	
	# Suppression des fichiers temporaires
	find . -type f -name "*.tmp" -delete 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	
	@echo "$(GREEN)✅ Nettoyage léger terminé$(NC)"

# Redémarre les services
restart:
	@echo "$(YELLOW)🔄 Redémarrage de PatentMuse...$(NC)"
	@make down
	@sleep 2
	@make start

# Test rapide des services
test:
	@echo "$(CYAN)🧪 Test des services PatentMuse...$(NC)"
	@make check-services

# Affiche l'aide avec style
help:
	@echo "$(GREEN)📖 Commandes PatentMuse disponibles$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "  $(GREEN)make start$(NC)      - $(CYAN)Lance tous les services et affiche les URLs$(NC)"
	@echo "  $(GREEN)make down$(NC)       - $(CYAN)Arrête tous les services$(NC)"
	@echo "  $(GREEN)make urls$(NC)       - $(CYAN)Affiche les URLs des services$(NC)"
	@echo "  $(GREEN)make test$(NC)       - $(CYAN)Vérifie le statut des services$(NC)"
	@echo "  $(GREEN)make logs$(NC)       - $(CYAN)Affiche les logs en temps réel$(NC)"
	@echo "  $(GREEN)make status$(NC)     - $(CYAN)Affiche le statut des conteneurs$(NC)"
	@echo "  $(GREEN)make rebuild$(NC)    - $(CYAN)Reconstruit et relance les services$(NC)"
	@echo "  $(GREEN)make restart$(NC)    - $(CYAN)Redémarre tous les services$(NC)"
	@echo "  $(GREEN)make clean$(NC)      - $(CYAN)Nettoyage sécurisé (préserve volumes)$(NC)"
	@echo "  $(GREEN)make clean-all$(NC)  - $(CYAN)Nettoyage complet + volumes (DANGER)$(NC)"
	@echo "  $(GREEN)make clean-light$(NC) - $(CYAN)Nettoyage léger (fichiers temporaires)$(NC)"
	@echo "  $(GREEN)make help$(NC)       - $(CYAN)Affiche cette aide$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

# Commande par défaut
.DEFAULT_GOAL := start