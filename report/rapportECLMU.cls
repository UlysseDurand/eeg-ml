\LoadClass[a4paper,12pt]{article}

\RequirePackage[french]{babel}
\RequirePackage[utf8]{inputenc}
\RequirePackage[section]{placeins}
\RequirePackage[T1]{fontenc}
\RequirePackage{fontspec}
\RequirePackage{biblatex}
\RequirePackage{mathtools}
\RequirePackage{siunitx}
\RequirePackage{float}
\RequirePackage{graphicx}
\RequirePackage[justification=centering]{caption}
\RequirePackage{subcaption}
\RequirePackage{wallpaper}
\RequirePackage{nomencl}
\RequirePackage{fancyhdr}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[left=2.5cm,right=2.5cm,top=2cm,bottom=3.5cm]{geometry}
\RequirePackage{cleveref}
\RequirePackage{tikz}

\usetikzlibrary{graphs, graphdrawing, positioning, arrows.meta, shapes.geometric, fit, calc}
\usegdlibrary{force}

\newcommand{\ttitle}[1]{\renewcommand{\ttitle}{#1}}
\newcommand{\UE}[1]{\renewcommand{\UE}{#1}}
\newcommand{\subject}[1]{\renewcommand{\subject}{#1}}
\newcommand{\tutor}[1]{\renewcommand{\tutor}{#1}}
\newcommand{\student}[1]{\renewcommand{\student}{#1}}

\renewcommand{\sectionmark}[1]{\markboth{#1}{}}
\newcommand{\makemargins}{
  \makenomenclature
  \pagestyle{fancy}
  \fancyheadoffset{1cm}
  \setlength{\headheight}{2cm}
  \lhead{
    \includegraphics[height=0.05\textheight]{assets/logos/logoECL.png}
    \hspace{1cm}
    \includegraphics[height=0.05\textheight]{assets/logos/logoMU.pdf}
  }
  \rhead{ \nouppercase{\leftmark} }
  \rfoot{\thepage}
  \cfoot{ }
  \lfoot{\ttitle}
}

\newcommand{\makefirstpage}{
  \begin{titlepage}

    \ThisLRCornerWallPaper{0.6}{assets/logos/SteinECL.jpg}
    \centering
    \includegraphics[height=0.1\textheight]{assets/logos/logoECL.png}\par\vspace{1cm}
    \includegraphics[height=0.1\textheight]{assets/logos/logoMU.pdf}\par\vspace{1cm}

    \vspace{1.5cm}
    {\scshape\Large \UE \\ \subject \par}
    \vspace{1cm}
    \rule{\linewidth}{0.2 mm} \\[0.4 cm]
    {\huge\bfseries \ttitle \par} \
    \rule{\linewidth}{0.2 mm} \\[1.5 cm]
    \vspace{1cm}

    \begin{minipage}{0.5\textwidth}
      \begin{flushleft} \large
        \emph{\textbf{\student}}\\
      \end{flushleft}
    \end{minipage}
    ~
    \begin{minipage}{0.4\textwidth}
      \begin{flushright} \large
        \emph{\textbf{\tutor}} \\
      \end{flushright}
    \end{minipage}\\[4cm]

    \vfill
    {\large \today\par}

  \end{titlepage}
  \clearpage

  \tableofcontents
  \clearpage
}

\newcommand{\insertfigure}[4]{
  \begin{figure}[H]
    \centering
    \includegraphics[keepaspectratio,#2]{#1}
    \caption{#3}
    \label{#4}
  \end{figure}
}

