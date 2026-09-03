(a)

Y. Sun, P.S.P. Tse Journal of Computational Physics 230 (2011) 20762094

2089

(b)

(c

(d)

(48)

(49)

(50)

Fig. 4. The dispersion contours with stepsizes t = 0.01, = 0.1 for Maxwell's equations (46) from (a) exact dispersion; (b) boxscheme; (c) symplectic method and (d) Yee's method. The constant contour values are {2.4, 6. . . ,24].}

\[
\colon \omega \in [ 2 , 4 , 6 , \dots , 2 4 ] .
\]

\[
\wp = \mathtt { t a n } ^ { - 1 } \left ( \frac { ( \nu _ { \mathtt { g } } ) _ { y } } { ( \nu _ { \mathtt { g } } ) _ { x } } \right ) , \quad | \nu _ { \mathtt { g } } | = \sqrt { ( \nu _ { \mathtt { g } } ) _ { x } ^ { 2 } + ( \nu _ { \mathtt { g } } ) _ { y } ^ { 2 } } .
\]

Substituting into (48) the vectors κ and $. ~ v _ { g }$ in polar coordinates (44), and let $\cdot a = | k | \Delta ,$ ,this yields the propagation angle $= \rho$ and
the propagation speed $| v _ { g } |$ in terms of a and $\dot { \theta }$
For example, $\varphi$ for the boxscheme is given by

\[
\wp = { \mathtt { t a l } } ^ { - 1 } ~ \left ( { \frac { \sin \left ( { \frac { 1 } { 2 } } ~ { \mathtt { S i n } } ( \theta ) \overline { a } \right ) ~ { \mathtt { C o s } } ^ { 3 } \left ( { \frac { 1 } { 2 } } ~ { \mathtt { C o s } } ( \theta ) \overline { a } \right ) } { \mathtt { C o s } ^ { 3 } ~ \left ( { \frac { 1 } { 2 } } ~ { \mathtt { S i n } } ( \theta ) \overline { a } \right ) ~ \sin \left ( { \frac { 1 } { 2 } } ~ { \mathtt { C o s } } ( \theta ) \overline { a } \right ) } } \right ) .
\]

Taking the Taylor expansion of this expression with respect to $a = 0$ vields,

\[
\wp \approx \theta - { \frac { 1 } { 1 2 } } ~ \sin ( 4 \theta ) a ^ { 2 } + O ( a ^ { 3 } ) .
\]

Similarly, the Taylor expansion of $| v _ { g } |$ at $\cdot ~ a = 0$ yields,

\[
| v _ { g } | \approx 1 + \biggl ( \frac { 1 } { 1 6 } ~ \cos ( 4 \theta ) - \frac { r ^ { 2 } } { 4 } + \frac { 3 } { 1 6 } \biggr ) a ^ { 2 } + O ( a ^ { 4 } ) ,
\]