(a)

Y. Sun, P.S.P. Tse /Journal of Computational Physics 230 (2011) 20762094

2089

(b)

(c)

(d)

(48)

(49)

(50)

Fig. 4. The dispersion contours with stepsizes t = 0.01, = 0.1 for Maxwell's equations (46) from (a) exact dispersion; (b) boxscheme; (c) symplectic method and (d) Yee's method. The constant contour values are [2,4, 6, .. ., 24].

\[
: \omega \in [ 2 , 4 , 6 , \dots , 2 4 ] .
\]

\[
\varphi = \mathsf { t a n } ^ { - 1 } \left ( \frac { ( v _ { \mathrm { g } } ) _ { y } } { ( v _ { \mathrm { g } } ) _ { x } } \right ) , \quad | v _ { \mathrm { g } } | = \sqrt { ( v _ { \mathrm { g } } ) _ { x } ^ { 2 } + ( v _ { \mathrm { g } } ) _ { y } ^ { 2 } } .
\]

Substituting into (48) the vectors κ and $. 2 g$ in polar coordinates (44), and let $\cdot \pmb { a } = | \pmb { k } | \pmb { \Delta } ,$ this yields the propagation angle $= 4 p$ and
the propagation speed $| v _ { g } |$ in terms of a and $1 \dot { o }$
For example, $\varphi$ for the boxscheme is given by

\[
\varphi = \mathsf { t a n } ^ { - 1 } ~ \left ( { \frac { \sin \left ( { \frac { 1 } { 2 } } ~ \sin ( \theta ) a \right ) \cos ^ { 3 } \left ( { \frac { 1 } { 2 } } ~ \cos ( \theta ) a \right ) } { \cos ^ { 3 } ~ \left ( { \frac { 1 } { 2 } } ~ \sin ( \theta ) a \right ) \sin \left ( { \frac { 1 } { 2 } } ~ \cos ( \theta ) a \right ) } } \right ) .
\]

Taking the Taylor expansion of this expression with respect to $a = 0$ yields,

\[
\varphi \approx \theta - { \frac { 1 } { 1 2 } } ~ \sin ( 4 \theta ) a ^ { 2 } + O ( a ^ { 3 } ) .
\]

Similarly, the Taylor expansion of $| t _ { E }$ at $\mathbf { \nabla } a = 0$ yields,

\[
| v _ { g } | \approx 1 + \biggl ( \frac { 1 } { 1 6 } ~ \cos ( 4 \theta ) - \frac { r ^ { 2 } } { 4 } \! + \! \frac { 3 } { 1 6 } \biggr ) a ^ { 2 } + O ( a ^ { 4 } ) ,
\]