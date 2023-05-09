ifeq ($(OS),Windows_NT)
	SHELL := pwsh.exe
else
	SHELL := pwsh
endif

.SHELLFLAGS := -NoProfile -Command

getcommitid: 
	$(eval COMMITID = $(shell git log -1 --pretty=format:"%H"))
getbranchname:
	$(eval BRANCH_NAME = $(shell (git branch --show-current ) -replace '/','.'))


.PHONY: all clean test lint act
all: test

REGISTRY_NAME := 
REPOSITORY_NAME := bmcclure89/
IMAGE_NAME := alpaca_lora
TAG := :latest
DOCKER_ENVS := -e BASE_MODEL=decapoda-research/llama-7b-hf -e OFFLOAD_FOLDER=/root/offload
DOCKER_VOLUMES := -v $${PWD}/mnt/cache:/root/.cache -v $${PWD}/mnt/offload:/root/offload
DOCKER_CMD := generate.py --load_8bit --lora_weights 'tloen/alpaca-lora-7b'

# Run Options
RUN_PORTS := -p 7860:7860

build: getcommitid getbranchname
	docker build -t $(REGISTRY_NAME)$(REPOSITORY_NAME)$(IMAGE_NAME)$(TAG) -t $(REGISTRY_NAME)$(REPOSITORY_NAME)$(IMAGE_NAME):$(BRANCH_NAME) -t $(REGISTRY_NAME)$(REPOSITORY_NAME)$(IMAGE_NAME):$(BRANCH_NAME)_$(COMMITID) .


run: build
	docker run -d $(RUN_PORTS) --env NVIDIA_DISABLE_REQUIRE=1 --gpus=all --shm-size 64g $(DOCKER_VOLUMES) $(REGISTRY_NAME)$(REPOSITORY_NAME)$(IMAGE_NAME)$(TAG) $(DOCKER_CMD)
run_shell:
	docker run -d $(RUN_PORTS) --env NVIDIA_DISABLE_REQUIRE=1 --gpus=all --shm-size 64g --entrypoint sleep $(DOCKER_VOLUMES) $(REGISTRY_NAME)$(REPOSITORY_NAME)$(IMAGE_NAME)$(TAG) infinity
run_it:
	docker run -it $(RUN_PORTS) --entrypoint sleep $(DOCKER_VOLUMES) $(REGISTRY_NAME)$(REPOSITORY_NAME)$(IMAGE_NAME)$(TAG) infinitiy

package:
	$$PackageFileName = "$$("$(IMAGE_NAME)" -replace "/","_").tar"; docker save $(REGISTRY_NAME)$(REPOSITORY_NAME)$(IMAGE_NAME)$(TAG) -o $$PackageFileName

size:
	docker inspect -f "{{ .Size }}" $(REGISTRY_NAME)$(REPOSITORY_NAME)$(IMAGE_NAME)$(TAG)
	docker history $(REGISTRY_NAME)$(REPOSITORY_NAME)$(IMAGE_NAME)$(TAG)

publish:
	docker login; docker push $(REGISTRY_NAME)$(REPOSITORY_NAME)$(IMAGE_NAME)$(TAG); docker logout
