# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# The package is installed into the build environment (see .readthedocs.yaml),
# so autodoc imports it without any path manipulation.

# -- Project information -----------------------------------------------------

project = 'phat-tails'
copyright = u'2021 Ryan Skene. All rights reserved.'
author = 'Ryan Skene'

# The full version, including alpha/beta/rc tags
from importlib.metadata import version as _pkg_version
release = _pkg_version('phat-tails')

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinx.ext.mathjax',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',     # NumPy-style docstring sections
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Notebooks are committed with their outputs, produced on fixed seeds and the
# bundled data. The docs build renders them and never re-executes them.
nbsphinx_execute = 'never'

# Code cells tagged `hide_input` (the plotting boilerplate) show their output
# only. nbsphinx has no option for this, so the tag is translated here into the
# marker nbconvert's templates already understand (what its
# TagRemovePreprocessor sets).
import nbsphinx  # noqa: E402

_export_notebook = nbsphinx.Exporter.from_notebook_node


def _export_hiding_tagged_inputs(self, nb, resources=None, **kwargs):
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'hide_input' in cell.metadata.get('tags', []):
            cell.metadata['transient'] = {'remove_source': True}
    return _export_notebook(self, nb, resources, **kwargs)


nbsphinx.Exporter.from_notebook_node = _export_hiding_tagged_inputs

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/rjskene/phat',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_edit_page_button': True,
    'path_to_docs': 'docs',
}
mathjax3_config = {
    'tex': {'displayMath': [],
        },
    'chtml': {
        'scale': 2,
        # 'mathmlSpacing': false,
        'displayAlign': 'center',
        'displayIndent': '0'
  }
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_additional_pages = {'index': 'intro.html'}

master_doc = 'contents'
