.PHONY: help pypi pypi-test docs coverage test clean

help:
	@echo "pypi - submit to PyPI server"
	@echo "pypi-test - submit to TestPyPI server"
	@echo "docs - generate Sphinx documentation"
	@echo "coverage - check code coverage"
	@echo "test - run unit tests"
	@echo "clean - remove artifacts"

pypi:
	python -m build
	twine upload dist/*

pypi-check:
	python -m build
	twine check dist/*

pypi-test:
	python -m build
	twine upload -r testpypi dist/*

docs:
	rm -f docs/calistar.rst
	sphinx-apidoc -o docs calistar
	cd docs/
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

coverage:
	coverage run --source=calistar -m pytest
	coverage report -m

test:
	pytest --cov=calistar/ --cov-report=xml

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	rm -f .coverage*
	rm -f coverage.xml
	rm -f docs/calib_*
	rm -f docs/target_*
	rm -f gaiaxp_6843672087120107264_0.jpg
	rm -f gaiaxp_6843672087120107264_spec.dat
	rm -f gaiaxp_6843672087120107264_cov.dat
	rm -f docs/gaiaxp_6843672087120107264_0.jpg
	rm -f docs/gaiaxp_6843672087120107264.dat
	rm -f calib_*
	rm -f target_*
	rm -rf .pytest_cache/
	rm -rf docs/_build/
	rm -rf docs/.ipynb_checkpoints/
	rm -rf build/
	rm -rf dist/
	rm -rf calistar.egg-info/
	rm -rf htmlcov/
	rm -rf .tox/
