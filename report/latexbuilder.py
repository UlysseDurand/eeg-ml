#!/usr/bin/env python3

import yaml
from jinja2 import Template
import sys

infile = sys.argv[1]
metafile = sys.argv[2]
outfile = sys.argv[3]
srcfolder = sys.argv[4]
bodytex = sys.argv[5]

with open(metafile, 'r') as f:
    variables = yaml.safe_load(f)
variables["srcfolder"] = srcfolder
variables["bodytex"] = bodytex

with open(infile, 'r') as f:
    template_content = f.read()

template = Template(template_content)
rendered_tex = template.render(**variables)

with open(outfile, 'w') as f:
    f.write(rendered_tex)

print(f"{outfile} generated successfully.")