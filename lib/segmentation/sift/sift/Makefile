# file:        Makefile
# author:      Andrea Vedaldi
# description: Build SIFT mex files

NAME=sift
VER=0.9.19

# --------------------------------------------------------------------
#                                                       Error messages
# --------------------------------------------------------------------

err_no_arch  =
err_no_arch +=$(shell echo "** Unknown host architecture '$(UNAME)'. This identifier"   1>&2)
err_no_arch +=$(shell echo "** was obtained by running 'uname -sm'. Edit the Makefile " 1>&2)
err_no_arch +=$(shell echo "** to add the appropriate configuration."                   1>&2)
err_no_arch +=Configuration failed

# --------------------------------------------------------------------
#                                                        Configuration
# --------------------------------------------------------------------

CFLAGS           += -I. -pedantic -Wall -g -O3
CFLAGS           += -Wno-variadic-macros
LDFLAGS          +=
MEX_CFLAGS        = CFLAGS='$$CFLAGS $(CFLAGS)'

# Determine on the flight the system we are running on
Darwin_PPC_ARCH             := mac
Darwin_Power_Macintosh_ARCH := mac
Darwin_i386_ARCH            := mci
Linux_i386_ARCH             := glx
Linux_i686_ARCH             := glx
Linux_x86_64_ARCH           := g64
Linux_unknown_ARCH          := glx

UNAME             := $(shell uname -sm)
ARCH              := $($(shell echo "$(UNAME)" | tr \  _)_ARCH)

mac_CFLAGS       := -faltivec
mac_MEX_CFLAGS   := CC='gcc' CXX='g++' LD='gcc'
mac_MEX_SUFFIX   := mexmac

mci_CFLAGS       :=
mci_MEX_CFLAGS   :=
mci_MEX_SUFFIX   := mexmaci

glx_CFLAGS       :=
glx_MEX_CFLAGS   :=
glx_MEX_SUFFIX   := mexglx

g64_CFLAGS       :=
g64_MEX_CFLAGS   :=
g64_MEX_SUFFIX   := mexa64

CFLAGS           += $($(ARCH)_CFLAGS)
MEX_SUFFIX       := $($(ARCH)_MEX_SUFFIX)
MEX_CFLAGS       += $($(ARCH)_MEX_CFLAGS)
DIST             := $(NAME)-$(VER)
BINDIST          := $(DIST)-bin

ifeq ($(ARCH),)
die:=$(error $(err_no_arch))
endif

# --------------------------------------------------------------------
#
# --------------------------------------------------------------------

src :=\
imsmooth.c \
siftlocalmax.c \
siftrefinemx.c \
siftormx.c \
siftdescriptor.c \
siftmatch.c

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

# --------------------------------------------------------------------
#                                                                 Dist
# --------------------------------------------------------------------

.PHONY: dist, bindist, clean, distclean, info, clean-$(NAME)

TIMESTAMP:
	echo "Version $(VER)"            > TIMESTAMP
	echo "Archive created on `date`" >>TIMESTAMP

VERSION: Makefile
	echo "$(VER)" > VERSION

clean:
	rm -f $(tgt)
	rm -f doc/*.log
	rm -f doc/*.aux
	rm -f doc/*.toc
	rm -f doc/*.blg
	rm -f doc/*.out
	rm -f .gdb_history

distclean: clean
	rm -f *.mexmac *.mexglx *.mexmaci *.dll
	rm -f `find . -name '.DS_Store'`
	rm -f `find . -name '*~'`
	rm -f doc/sift.pdf doc/index.html doc/default.css
	rm -f TIMESTAMP VERSION
	rm -rf $(NAME) $(NAME)-*

info:
	@echo ARCH=$(ARCH)

clean-$(NAME) : 
	rm -rf $(NAME)

$(NAME): clean-$(NAME) TIMESTAMP VERSION
	git archive --prefix=$(NAME)/ HEAD | tar xvf -
	cp TIMESTAMP $(NAME)
	cp VERSION $(NAME)

dist: clean-$(NAME) $(NAME)
	COPYFILE_DISABLE=1                                           \
	COPY_EXTENDED_ATTRIBUTES_DISABLE=1                           \
	tar czvf $(DIST).tar.gz $(NAME)

bindist: all doc $(NAME)
	cp -v `find -E . -maxdepth 1 -regex '.*\.mex.*$$'` $(NAME)
	cp doc/sift.pdf doc/index.html doc/default.css $(NAME)/doc
	COPYFILE_DISABLE=1                                           \
	COPY_EXTENDED_ATTRIBUTES_DISABLE=1                           \
	tar czvf $(BINDIST).tar.gz                                   \
	    $(NAME)

