#!/usr/bin/perl -w 
## file:        sift_gendoc.pl
## author:      Andrea Vedaldi
## description: Summarize MATLAB M-Files docs

# AUTORIGHTS
# Copyright (c) 2006 The Regents of the University of California.
# All Rights Reserved.
# 
# Created by Andrea Vedaldi
# UCLA Vision Lab - Department of Computer Science
# 
# Permission to use, copy, modify, and distribute this software and its
# documentation for educational, research and non-profit purposes,
# without fee, and without a written agreement is hereby granted,
# provided that the above copyright notice, this paragraph and the
# following three paragraphs appear in all copies.
# 
# This software program and documentation are copyrighted by The Regents
# of the University of California. The software program and
# documentation are supplied "as is", without any accompanying services
# from The Regents. The Regents does not warrant that the operation of
# the program will be uninterrupted or error-free. The end-user
# understands that the program was developed for research purposes and
# is advised not to rely exclusively on the program for any reason.
# 
# This software embodies a method for which the following patent has
# been issued: "Method and apparatus for identifying scale invariant
# features in an image and use of same for locating an object in an
# image," David G. Lowe, US Patent 6,711,293 (March 23,
# 2004). Provisional application filed March 8, 1999. Asignee: The
# University of British Columbia.
# 
# IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
# ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF
# CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
# BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE
# MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

# Debugging level
$verb = 0 ;

# PDF document location
$pdfdoc = 'sift.pdf' ;

# This is the queue of directories to process.
# It gets filled recursively.
@dir_fifo = ('.') ;

# This will hold an entry for each m-file found
%mfiles = () ;

# This will hold an entry for each module found 
%modules = () ;

# #ARGV is the index of the last element, which is 0
if ($#ARGV == 0) {
    open(FOUT,">$ARGV[0]") ;
    print STDERR "Writing to file '$ARGV[0]'.\n" if $verb ;
} else {
    *FOUT= *STDOUT; 
    print STDERR "Using standard output.\n" if $verb ; 
}

# Each module is a subdirectory. The subdirectory is used
# as the module ID.

while ($module_path = shift(@dir_fifo)) { 
    print STDERR "=> '$module_path'\n" if $verb ;
    
    # get a first version of the module name
    $module_path =~ m/.*\/([^\\]+)$/ ;
    $module_name = $1 ;

    # start a new module    
    $module = {
        'id'          => $module_path,
        'path'        => $module_path,
        'name'        => $module_name,
        'mfiles'      => [],
        'description' => ""
    } ;
    
    # .................................................................
    opendir(DIRHANDLE, $module->{'path'}) ;
    FILE: foreach (sort readdir(DIRHANDLE)) {
        $name = $_ ;
        $path = $module->{'path'} . "/" . $_ ;

        # push if a directory
        $_ = $path ;
        if (-d) {
            next FILE if /(\.$|\.\.$)/ ;
            push( @dir_fifo, "$_" ) ;
            next FILE ;
        }
        
        # parse if .m and not test_
        next FILE unless (/.+\.m$/) ;
        next FILE if (/test_.x*/) ;
        $name =~ m/(.*).m/ ;
        $name = $1 ;
        print STDERR "  . m-file '$name'\n" if $verb ;
        my ($brief,$description) = get_comment($path) ;

        # topic description?
        if (/overview/) {
            print STDERR "    * module description\n" if $verb ;
            $module->{'id'}          = $name ;
            $module->{'name'}        = $brief ;
            $module->{'description'} = $description ;
            next FILE ;
        }

        # use names as IDs
        $id = $name ;
        $id =~ tr/A-Z/a-z/ ;

        # create a new mfile object
        $mfile = {
            'id'          => $id,
            'path'        => $path,
            'name'        => $name,
            'brief'       => $brief,
            'module'      => $module,
            'description' => $description
        } ;
        
        # add a reference to this mfile
        # object to the global mfile list
        $mfiles{$id} = $mfile ;

        # add a reference to this mfile
        # object to the current module
        push( @{$module->{'mfiles'}}, $mfile) ;
    }
    closedir(DIRHANDLE) ;
    # ................................................................

    # add a reference to the current module to the global
    # module list
    $modules{$module->{'id'}} = $module ;
}

# ....................................................................
#                                                  write documentation
# ....................................................................

print FOUT <<EOF;
<html>
  <head>
    <link href="default.css" rel="stylesheet" type="text/css"/>
  </head>
  <body>
EOF

# sort modules by path
sub criterion { $modules{$a}->{'path'} cmp $modules{$b}->{'path'} ; }

MODULE:
foreach $id ( sort criterion keys %modules ) {
    my $module = $modules{$id} ;   
    my $rich_description = make_rich($module->{'description'}) ;

    next MODULE if $#{$module->{'mfiles'}} < 0 and length($rich_description) == 0;

    print FOUT <<EOF;
    <div class='module' id='$module->{"id"}'>
      <h1>$module->{'name'}</h1>
      <div class='index'>
        <h1>Module contents</h1>
        <ul>
EOF
    foreach( @{$module->{'mfiles'}} ) {
        print FOUT "        <li><a href='#$_->{'id'}'>" 
                 . "$_->{'name'}</a></li>\n" ;
    }
    print FOUT <<EOF;
        </ul>
      </div>
      <pre>
      $rich_description
      </pre>
      <div class="footer">
      </div>
    </div>
EOF
}

foreach $id (sort keys %mfiles) {
    my $mfile = $mfiles{$id} ;
    my $rich_description = make_rich($mfile->{'description'}) ;

    print FOUT <<EOF;
    <div class="mfile" id='$mfile->{"id"}'>
      <h1>
       <span class="name">$mfile->{"name"}</span>
       <span class="brief">$mfile->{"brief"}</span>
      </h1>
      <pre>
$rich_description
      </pre>
      <div class="footer">
        <a href="#$mfile->{'module'}->{'id'}">
        $mfile->{'module'}->{'name'}
        </a>
      </div>
    </div>
EOF
}

print FOUT "</body></html>" ;

# Close file
close FOUT ;


# -------------------------------------------------------------------------
sub get_comment {
# -------------------------------------------------------------------------
    local $_ ;
    my $full_name = $_[0] ;
    my @comment = () ;
    
    open IN,$full_name ;
  SCAN: 
    while( <IN> ) {
        next if /^function/ ;
        last SCAN unless ( /^%/ ) ;
        push(@comment, substr("$_",1)) ;
    }
    close IN ;

    my $brief = "" ;
    if( $#comment >= 0 && $comment[0] =~ m/^\s*\w+\s+(.*)$/ ) {
        $brief = $1 ;
        splice (@comment, 0, 1) ;
    }

    # from the first line
    return ($brief, join("",@comment)) ;
}

# -------------------------------------------------------------------------
sub make_rich {
# -------------------------------------------------------------------------
    local $_ = $_[0] ;
    s/([A-Z]+[A-Z0-9_]*)\(([^\)]*)\)/${\make_link($1,$2)}/g ;
    s/PDF:([A-Z0-9_\-:.]+[A-Z0-9])/${\make_pdf_link($1)}/g ;
    return $_ ;
}

# -------------------------------------------------------------------------
sub make_link {
# -------------------------------------------------------------------------
    local $_ ;
    my $name = $_[0] ;
    my $arg  = $_[1] ;
    my $id   = $name ;

    # convert name to lower case and put into $_
    $id =~ tr/A-Z/a-z/ ;
    
    # get mfile
    my $mfile = $mfiles{$id} ;
    my $module = $modules{$id} ;
    
    # return as appropriate    
    if($mfile) {
        return "<a href='#$id'>" . $name . "</a>" . "(" . $arg . ")" ;
    } elsif($module) {
      return "<a class='module' href='#$id'>" . $name . 
          "</a>" . "(" . $arg . ")" ;
    } else {
        return $name . "(" . $arg .")" ;
    }
}


# -------------------------------------------------------------------------
sub make_pdf_link {
# -------------------------------------------------------------------------
    local $_ ;
    my $name = $_[0] ;
    my $id   = $name ;

    # convert name to lower case and put into $_
    $id =~ tr/A-Z/a-z/ ;
    
    return "<a href='${pdfdoc}#$id'>PDF:$1</a>" ;
}
