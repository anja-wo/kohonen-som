.PHONY: test

style:
	black src test
	isort src test

quality:
	mypy src test
	black --check src test
	isort --check-only src test
	flake8 src test

test:
	python -m pytest test

testcov:
	python -m pytest test --cov=src

package: test
		poetry build

upload: package
		poetry publish -r nexus-hosted
