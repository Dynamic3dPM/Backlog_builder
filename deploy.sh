#!/bin/bash

# Backlog Builder AI Services Deployment Script
# This script helps deploy the AI services pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
WITH_GPU=false
BUILD_ONLY=false
SKIP_TESTS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --gpu)
            WITH_GPU=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment ENV    Set environment (development|production) [default: development]"
            echo "  --gpu                    Include GPU services for local AI processing"
            echo "  --build-only            Only build images, don't start services"
            echo "  --skip-tests            Skip running tests before deployment"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🚀 Backlog Builder AI Services Deployment${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}GPU Support: ${WITH_GPU}${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

# Check NVIDIA Docker (if GPU enabled)
if [ "$WITH_GPU" = true ]; then
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        echo -e "${RED}❌ NVIDIA Docker runtime is not available${NC}"
        echo -e "${YELLOW}💡 Install nvidia-container-toolkit or disable GPU support${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ NVIDIA Docker runtime available${NC}"
fi

echo -e "${GREEN}✅ Prerequisites met${NC}"

# Environment setup
echo -e "${YELLOW}🔧 Setting up environment...${NC}"

if [ "$ENVIRONMENT" = "production" ]; then
    ENV_FILE=".env.production"
    COMPOSE_FILE="docker-compose.prod.yml"
else
    ENV_FILE=".env.example"
    COMPOSE_FILE="docker-compose.yml"
fi

# Copy environment file if it doesn't exist
if [ ! -f "ai-services/.env" ]; then
    if [ -f "ai-services/$ENV_FILE" ]; then
        cp "ai-services/$ENV_FILE" "ai-services/.env"
        echo -e "${GREEN}✅ Environment file created from $ENV_FILE${NC}"
        echo -e "${YELLOW}⚠️  Please update ai-services/.env with your API keys${NC}"
    else
        echo -e "${RED}❌ Environment file ai-services/$ENV_FILE not found${NC}"
        exit 1
    fi
fi

# Create necessary directories
mkdir -p uploads/audio
mkdir -p logs
mkdir -p monitoring/prometheus
mkdir -p monitoring/grafana/provisioning
mkdir -p monitoring/grafana/dashboards

echo -e "${GREEN}✅ Environment setup complete${NC}"

# Build Docker images
echo -e "${YELLOW}🔨 Building Docker images...${NC}"

COMPOSE_PROFILES=""
if [ "$WITH_GPU" = true ]; then
    COMPOSE_PROFILES="--profile gpu"
fi

docker-compose -f $COMPOSE_FILE $COMPOSE_PROFILES build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker images built successfully${NC}"
else
    echo -e "${RED}❌ Failed to build Docker images${NC}"
    exit 1
fi

if [ "$BUILD_ONLY" = true ]; then
    echo -e "${GREEN}✅ Build complete. Images ready for deployment.${NC}"
    exit 0
fi

# Run tests (if not skipped)
if [ "$SKIP_TESTS" = false ]; then
    echo -e "${YELLOW}🧪 Running tests...${NC}"
    
    # Start test dependencies
    docker-compose -f $COMPOSE_FILE up -d redis
    
    # Wait for Redis to be ready
    echo "Waiting for Redis to be ready..."
    until docker-compose -f $COMPOSE_FILE exec redis redis-cli ping; do
        sleep 1
    done
    
    # Run backend tests
    if [ -f "backend/package.json" ]; then
        echo "Running backend tests..."
        cd backend
        npm test || {
            echo -e "${RED}❌ Backend tests failed${NC}"
            exit 1
        }
        cd ..
    fi
    
    # Run AI services tests
    echo "Running AI services health checks..."
    
    # Start AI services for testing
    docker-compose -f $COMPOSE_FILE $COMPOSE_PROFILES up -d ai-stt-cloud ai-llm-cloud
    
    # Wait for services to be ready
    sleep 30
    
    # Test cloud STT service
    if docker-compose -f $COMPOSE_FILE exec ai-stt-cloud curl -f http://localhost:8002/health; then
        echo -e "${GREEN}✅ Cloud STT service health check passed${NC}"
    else
        echo -e "${RED}❌ Cloud STT service health check failed${NC}"
        exit 1
    fi
    
    # Test cloud LLM service
    if docker-compose -f $COMPOSE_FILE exec ai-llm-cloud curl -f http://localhost:8004/health; then
        echo -e "${GREEN}✅ Cloud LLM service health check passed${NC}"
    else
        echo -e "${RED}❌ Cloud LLM service health check failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All tests passed${NC}"
fi

# Deploy services
echo -e "${YELLOW}🚀 Deploying services...${NC}"

# Stop any existing services
docker-compose -f $COMPOSE_FILE down

# Start services
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f $COMPOSE_FILE $COMPOSE_PROFILES up -d
else
    docker-compose -f $COMPOSE_FILE $COMPOSE_PROFILES up -d
fi

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 60

# Health checks
echo -e "${YELLOW}🏥 Running health checks...${NC}"

# Check backend
if curl -f http://localhost:3000/health &> /dev/null; then
    echo -e "${GREEN}✅ Backend service is healthy${NC}"
else
    echo -e "${RED}❌ Backend service health check failed${NC}"
fi

# Check Redis
if docker-compose -f $COMPOSE_FILE exec redis redis-cli ping &> /dev/null; then
    echo -e "${GREEN}✅ Redis is healthy${NC}"
else
    echo -e "${RED}❌ Redis health check failed${NC}"
fi

# Check cloud STT
if curl -f http://localhost:8002/health &> /dev/null; then
    echo -e "${GREEN}✅ Cloud STT service is healthy${NC}"
else
    echo -e "${RED}❌ Cloud STT service health check failed${NC}"
fi

# Check cloud LLM
if curl -f http://localhost:8004/health &> /dev/null; then
    echo -e "${GREEN}✅ Cloud LLM service is healthy${NC}"
else
    echo -e "${RED}❌ Cloud LLM service health check failed${NC}"
fi

# Check GPU services (if enabled)
if [ "$WITH_GPU" = true ]; then
    if curl -f http://localhost:8001/health &> /dev/null; then
        echo -e "${GREEN}✅ Local STT service is healthy${NC}"
    else
        echo -e "${YELLOW}⚠️  Local STT service health check failed${NC}"
    fi
    
    if curl -f http://localhost:8003/health &> /dev/null; then
        echo -e "${GREEN}✅ Local LLM service is healthy${NC}"
    else
        echo -e "${YELLOW}⚠️  Local LLM service health check failed${NC}"
    fi
fi

# Display service information
echo ""
echo -e "${GREEN}🎉 Deployment complete!${NC}"
echo ""
echo -e "${BLUE}📊 Service URLs:${NC}"
echo -e "  Backend API:      http://localhost:3000"
echo -e "  Cloud STT:        http://localhost:8002"
echo -e "  Cloud LLM:        http://localhost:8004"

if [ "$WITH_GPU" = true ]; then
    echo -e "  Local STT:        http://localhost:8001"
    echo -e "  Local LLM:        http://localhost:8003"
fi

echo -e "  Redis:            localhost:6379"

if [ "$ENVIRONMENT" = "production" ]; then
    echo -e "  Prometheus:       http://localhost:9090"
    echo -e "  Grafana:          http://localhost:3001"
    echo -e "  PostgreSQL:       localhost:5432"
fi

echo ""
echo -e "${BLUE}📚 Next steps:${NC}"
echo -e "  1. Update ai-services/.env with your API keys"
echo -e "  2. Test the API endpoints using the provided examples"
echo -e "  3. Monitor services using docker-compose logs -f"

if [ "$ENVIRONMENT" = "production" ]; then
    echo -e "  4. Configure SSL certificates in nginx/ssl/"
    echo -e "  5. Set up database backups"
    echo -e "  6. Configure monitoring alerts"
fi

echo ""
echo -e "${GREEN}✅ Deployment successful! Happy coding! 🚀${NC}"
