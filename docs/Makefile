# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

SPHINXAPIDOC  = sphinx-apidoc
PACKAGEDIR    = ../../

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help apidoc ga noga Makefile

# Compile API doc.
apidoc:
	@echo "Compiling API documentation..."; \
	$(SPHINXAPIDOC) -M -f -o "$(SOURCEDIR)" "$(PACKAGEDIR)"

# Enable Google analytics.
ga:
	@echo "Enabling Google analytics..."; \
	sed -i "s/googleanalytics_enabled = .*/googleanalytics_enabled = True/" source/conf.py

# Disable Google analytics.
noga:
	@echo "Disabling Google analytics..."; \
	sed -i "s/googleanalytics_enabled = .*/googleanalytics_enabled = False/" source/conf.py

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
