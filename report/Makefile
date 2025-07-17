TEMPLATEFILE = template.tex
TEXFILE = main
OUTDIR = build
CLSFILE = rapportECLMU.cls
METAFILE = meta.yaml
SRCFOLDER = src
BODYTEX = body.tex
DEPFILES = $(wildcard $(SRCFOLDER)/*)

#FAST= -usepretex -bibtex-
FAST=

.PHONY: clean

$(OUTDIR)/$(TEXFILE).pdf: $(OUTDIR)/$(TEXFILE).tex 
	@mkdir -p $(OUTDIR)
	latexmk -f -lualatex -silent $(FAST) -outdir=$(OUTDIR) $(OUTDIR)/$(TEXFILE).tex

$(OUTDIR)/$(TEXFILE).tex: $(TEMPLATEFILE) $(METAFILE) $(CLSFILE) $(DEPFILES)
	@mkdir -p $(OUTDIR)
	./latexbuilder.py $(TEMPLATEFILE) $(METAFILE) $(OUTDIR)/$(TEXFILE).tex $(SRCFOLDER) $(BODYTEX)

clean:
	latexmk -C -outdir=$(OUTDIR)
	@rm -f $(OUTDIR)/$(TEXFILE).pdf