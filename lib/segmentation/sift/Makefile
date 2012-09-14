# file:        Makefile
# author:      Andrea Vedaldi
# description: Build SIFT mex files

DIST=sift
VER=0.9.7

src :=\
imsmooth.c \
siftlocalmax.c \
siftrefinemx.c \
siftormx.c \
siftdescriptor.c \
siftmatch.c

# --------------------------------------------------------------------
#
# --------------------------------------------------------------------

# Determine on the flight the system we are running on
Darwin_ARCH := mac
Linux_ARCH  := glx
ARCH := $($(shell uname)_ARCH)

mac_CFLAGS       := -O -I. -faltivec -pedantic -Wall
mac_MEX_CFLAGS   := CFLAGS='$$CFLAGS $(mac_CFLAGS)'
mac_MEX_SUFFIX   := mexmac

glx_CFLAGS       := -O -I. -pedantic -Wall
glx_MEX_CFLAGS   := CFLAGS='$$CFLAGS $(glx_CFLAGS)'
glx_MEX_SUFFIX   := mexglx

MEX_SUFFIX   := $($(ARCH)_MEX_SUFFIX)
MEX_CFLAGS   := $($(ARCH)_MEX_CFLAGS)

BINDIST=$(DIST)-$(ARCH)-$(VER)

# --------------------------------------------------------------------
#
# --------------------------------------------------------------------

%.$(MEX_SUFFIX) : %.c
	mex -I. $(MEX_CFLAGS) $< -o $*

tgt = $(src:.c=.$(MEX_SUFFIX))

.PHONY: all
all: $(tgt)

# PDF documentation
.PHONY: doc
doc: doc/sift.pdf doc/index.html doc/default.css

doc/index.html : $(wildcard *.m)
	mdoc --output=$@ . -x extra

doc/default.css : sift_gendoc.css
	ln -s ../sift_gendoc.css doc/default.css	

doc/sift.pdf : doc/*.tex doc/*.bib doc/figures/*
	cd doc ; \
	for k in 1 2 3 ; \
	do \
	  pdflatex -file-line-error-style -interaction batchmode \
	    sift.tex ; \
	  if test "$$k" = '1' ; \
	  then \
	    bibtex sift.aux ; \
	  fi ; \
	done

.PHONY: clean
clean:
	rm -f $(tgt)
	rm -f .DS_Store .gdb_history
	rm -f doc/*.log
	rm -f doc/*.aux
	rm -f doc/*.toc
	rm -f doc/*.bbl
	rm -f doc/*.blg
	rm -f doc/*.out

.PHONY: distclean
distclean: clean
	rm -f *.mexmac *.mexglx
	rm -f $(DIST)-*.zip
	rm -rf $(BINDIST)
	find . -name "*~"        -exec rm -f \{\} \;
	find . -name ".DS_Store" -exec rm -f \{\} \;
	rm -f doc/sift.pdf doc/index.html doc/default.css

.PHONY: zip
zip: dist

.PHONY: dist
dist: distclean
	echo Version $(VER) >TIMESTAMP
	echo Archive created on `date` >>TIMESTAMP
	echo On `uname -a` >>TIMESTAMP
	cd .. ; zip -q -r sift/$(DIST)-$(VER).zip sift -x "sift/extra/*"


.PHONY: bindist
bindist: all
	test -e $(BINDIST) || mkdir $(BINDIST)
	cp *.$(MEX_SUFFIX) $(BINDIST)
	cd $(BINDIST) ; strip -S *.$(MEX_SUFFIX)
	zip -q $(BINDIST).zip $(BINDIST)
