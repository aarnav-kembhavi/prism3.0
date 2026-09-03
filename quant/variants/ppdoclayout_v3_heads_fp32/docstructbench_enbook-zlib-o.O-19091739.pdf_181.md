Exat C nt.

Mo We are asked to find the magnetic field due to a simple current distribution, so this example is a typical problem for which the Biot-Savart law is appropriate. We must find the field contribution from a small element of current and then integrate over the current

AnalyseLet's start by considering a length element dš located a distance r from P. The
direction of the magnetic field at point P due to the current in this element is out of the
page because $d \bar { \mathbf { s } } \times \tilde { \hat { \mathbf { r } } } ~ .$ is out of the page. In fact, because all the current elements I ds lie in
the plane of the page, they all produce a magnetic field directed out of the page at point P
Therefore, the direction of the magnetic field at point P is out of the page and we need only
find the magnitude of the field. We place the origin at O and let point P be along the positive
y axis, with k being a unit vector pointing out of the page.

\[
) \theta _ { 2 }
\]

radians.

\[
\left ( { \frac { \pi } { 2 } } - \theta \right )
\]

CHAPTER 29 MAGNETIC FIELDS 819

From the geometry in Figure 29.3a, we can see that the angle between the vectors d š and r is radians.

\[
d ~ { \vec { \mathbf { s } } } \times { \dot { \mathbf { r } } } = { \big | } d ~ { \vec { \mathbf { s } } } \times { \dot { \mathbf { r } } } ] { \hat { \mathbf { k } } } = { \Bigg [ } d x ~ \sin \! \left ( { \frac { \pi } { 2 } } - \theta \right ) { \Bigg ] } { \dot { \mathbf { k } } } = ( d x ~ \cos \theta ) { \dot { \mathbf { k } } }
\]

Evaluate the cross product in the Biot-Savart law:

\[
d \vec { \mathbf { B } } = ( d B ) \hat { \mathbf { k } } = \frac { \mu _ { 0 } t } { 2 } \frac { d x \cos \theta } { \pi } \hat { \mathbf { k } }
\]

\[
4 \pi
\]

Substitute into Equation 29.1:

From the geometry in Figure 29.3a, express r in terms of :

(1)

(Example 29.1) (a) A thin,
straight wire carrying a
current I (b) The angles θ, and
$B _ { z }$ are used for determining
the net field.

(2)

(3)

(4)

\[
r = { \frac { a } { r } }
\]

\[
\cos \theta
\]

Notice that tan ${ } _ { 1 } \theta = - x / a$ from the right triangle in Figure 29.3a (the negative sign is necessary because ds is located at a
negative value of x) and solve for x:

\[
a \tan \theta
\]

Find the differential dx:

\[
d x = - a ~ s e c ^ { 2 } \theta ~ d \theta = - { \frac { a ~ d \theta } { \cos ^ { 2 } \theta } }
\]

Substitute Equations (2) and (3) into the magnitude of the field from Equation (1):

\[
d B = - { \frac { \mu _ { 0 } I } { 4 \pi } } \! \left ( { \frac { a ~ d \theta } { \cos ^ { 2 } \theta } } \right ) \! \left ( { \frac { \cos ^ { 2 } \theta } { a ^ { 2 } } } \right ) \! \cos \theta = - { \frac { \mu _ { 0 } I } { 4 \pi a } } \! \cos \theta ~ d \theta
\]

\[
1 ( 4 )
\]

Integrate Equation (4) over all length elements on the wire, where the subtending angles range from , to 2 as defined in Figure 29.3b: B = 4aJa01 cos d =4a1(sin 1 sin 2) (29.4)

\[
B = - \frac { \mu _ { 0 } I } { 4 \pi a } \! \int _ { \theta _ { 0 } } ^ { \theta _ { 2 } } \! \cos \theta d \theta = \frac { \mu _ { 0 } I } { 4 \pi a } ( \sin \theta _ { 1 } - \sin \theta _ { 2 } )
\]

Check the dimensions, noting that the quantity in brackets is dimensionless:

\[
[ \mathrm { M Q ~ ^ 1 T ^ 1 ] = [ M L Q ^ { - 2 } ] [ Q T ^ { ~ 1 } ] / [ L ] = [ M Q ~ ^ 1 T ^ 1 ] ~ \widehat \otimes }
\]

(B) Find an expression for the field at a point near a very long current-carrying wire.

## Solution

We can use Equation 29.4 to find the magnetic field of any straight current-carrying wire if we know the geometry and
hence the angles $\theta _ { 1 }$ anc $1 \theta _ { 2 }$ . If the wire in Figure 29.3b becomes infinitely long, we see that $\theta _ { _ 1 } = \pi / 2 ~ .$ and $1 ~ \tilde { \theta _ { 2 } } = - \pi / 2$ for