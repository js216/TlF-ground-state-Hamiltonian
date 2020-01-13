import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

####################################
### Units and constants
####################################

Jmax = 6      # max J value in Hamiltonian
I_Tl = 1/2    # I1 in Ramsey's notation
I_F = 1/2     # I2 in Ramsey's notation

# TlF constants. Data from D.A. Wilkening, N.F. Ramsey,
# and D.J. Larson, Phys Rev A 29, 425 (1984). Everything in Hz.

Brot = 6689920000
c1 = 126030.0
c2 = 17890.0
c3 = 700.0
c4 = -13300.0

D_TlF = 4.2282 * 0.393430307 *5.291772e-9/4.135667e-15 # [Hz/(V/cm)]

# Constants from Wilkening et al, in Hz/Gauss, for 205Tl

mu_J = 35
mu_Tl = 1240.5
mu_F = 2003.63


####################################
### Representing the states
####################################

class BasisState:
    """Class to represent a quantum basis state.

    A state, in general, can be written as a weighted superposition of the basis
    states. We work in the basis $|J, m_J, I_1, m_1, I_2, m_2\rangle$.

    The operations we can define on the basis states are:

        - construction: e.g. calling `BasisState(QN)` creates a basis state with
          quantum numbers `QN = (J, mJ, I1, m1, I2, m2)`;
        - equality testing;
        - inner product, returning either 0 or 1;
        - superposition and scalar multiplication, returning a `State` object
        - a convenience function to print out all quantum numbers
    """
    # constructor
    def __init__(self, J, mJ, I1, m1, I2, m2):
        self.J, self.mJ  = J, mJ
        self.I1, self.m1 = I1, m1
        self.I2, self.m2 = I2, m2

    # equality testing
    def __eq__(self, other):
        return self.J==other.J and self.mJ==other.mJ \
                    and self.I1==other.I1 and self.I2==other.I2 \
                    and self.m1==other.m1 and self.m2==other.m2

    # inner product
    def __matmul__(self, other):
        if self == other:
            return 1
        else:
            return 0

    # superposition: addition
    def __add__(self, other):
        if self == other:
            return State([ (2,self) ])
        else:
            return State([ (1,self), (1,other) ])

    # superposition: subtraction
    def __sub__(self, other):
        return self + -1*other

    # scalar product (psi * a)
    def __mul__(self, a):
        return State([ (a, self) ])

    # scalar product (a * psi)
    def __rmul__(self, a):
        return self * a
    
    def print_quantum_numbers(self):
        print( self.J,"%+d"%self.mJ,"%+0.1f"%self.m1,"%+0.1f"%self.m2 )

class State:
    """Class to represent a superposition state.

    A general state `State` can have any number of components, so let's
    represent it as an list of pairs `(amp, psi)`, where `amp` is the relative
    amplitude of a component, and `psi` is a basis state. The same component
    must not appear twice on the list.

    There are three operations we can define on the states:

    - construction
    - superposition: concatenate component arrays and return a `State`
    - scalar multiplication `a * psi` and `psi * a`, division, negation
    - component-wise inner product `psi1 @ psi2`, where `psi1` is a bra, and
      `psi2` a ket, returning a complex number

    In addition, I define an iterator method to loop through the components,
    and the `__getitem__()` method to access the components (which are not
    necessarily in any particular order!). See
    [Classes/Iterators](https://docs.python.org/3/tutorial/classes.html#iterators)
    for details.
    """
    # constructor
    def __init__(self, data=[], remove_zero_amp_cpts=True):
        # check for duplicates
        for i in range(len(data)):
            amp1,cpt1 = data[i][0], data[i][1]
            for amp2,cpt2 in data[i+1:]:
                if cpt1 == cpt2:
                    raise AssertionError("duplicate components!")
        # remove components with zero amplitudes
        if remove_zero_amp_cpts:
            self.data = [(amp,cpt) for amp,cpt in data if amp!=0]
        else:
            self.data = data
        # for iteration over the State
        self.index = len(self.data)

    # superposition: addition
    # (highly inefficient and ugly but should work)
    def __add__(self, other):
        data = []
        # add components that are in self but not in other
        for amp1,cpt1 in self.data:
            only_in_self = True
            for amp2,cpt2 in other.data:
                if cpt2 == cpt1:
                    only_in_self = False
            if only_in_self:
                data.append((amp1,cpt1))
        # add components that are in other but not in self
        for amp1,cpt1 in other.data:
            only_in_other = True
            for amp2,cpt2 in self.data:
                if cpt2 == cpt1:
                    only_in_other = False
            if only_in_other:
                data.append((amp1,cpt1))
        # add components that are both in self and in other
        for amp1,cpt1 in self.data:
            for amp2,cpt2 in other.data:
                if cpt2 == cpt1:
                    data.append((amp1+amp2,cpt1))
        return State(data)
                
    # superposition: subtraction
    def __sub__(self, other):
        return self + -1*other

    # scalar product (psi * a)
    def __mul__(self, a):
        return State( [(a*amp,psi) for amp,psi in self.data] )

    # scalar product (a * psi)
    def __rmul__(self, a):
        return self * a
    
    # scalar division (psi / a)
    def __truediv__(self, a):
        return self * (1/a)
    
    # negation
    def __neg__(self):
        return -1.0 * self
    
    # inner product
    def __matmul__(self, other):
        result = 0
        for amp1,psi1 in self.data:
            for amp2,psi2 in other.data:
                result += amp1.conjugate()*amp2 * (psi1@psi2)
        return result

    # iterator methods
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]
    
    # direct access to a component
    def __getitem__(self, i):
        return self.data[i]


####################################
### Operators in Python
####################################

# Define QM operators as Python functions that take `BasisState` objects, and
# return `State` objects. Since we are interested in finding matrix elements, we
# only need the action of operators on the basis states (but it'd be easy to
# generalize using a `for` loop).
# 
# The easiest operators to define are the diagonal ones $J^2, J_z, I_{1z},
# I_{2z}$, which just multiply the state by their eigenvalue:

def J2(psi):
    return State([(psi.J*(psi.J+1),psi)])

def Jz(psi):
    return State([(psi.mJ,psi)])

def I1z(psi):
    return State([(psi.m1,psi)])

def I2z(psi):
    return State([(psi.m2,psi)])

# The other angular momentum operators we can obtain through the ladder operators
# 
# $$ J_\pm=J_x\pm iJ_y. $$
# 
# These are defined through their action on the basis states as (Sakurai eqns
# 3.5.39-40)
# 
# $$ J_\pm|J,m\rangle=\sqrt{(j\mp m)(j\pm m+1)}|jm\pm1\rangle. $$
# 
# Similarly, $I_{1\pm},I_{2\pm}$ act on the $|I_1,m_1\rangle$ and
# $|I_2,m_2\rangle$ subspaces in the same way.

def Jp(psi):
    amp = sqrt((psi.J-psi.mJ)*(psi.J+psi.mJ+1))
    ket = BasisState(psi.J, psi.mJ+1, psi.I1, psi.m1, psi.I2, psi.m2)
    return State([(amp,ket)])

def Jm(psi):
    amp = sqrt((psi.J+psi.mJ)*(psi.J-psi.mJ+1))
    ket = BasisState(psi.J, psi.mJ-1, psi.I1, psi.m1, psi.I2, psi.m2)
    return State([(amp,ket)])

def I1p(psi):
    amp = sqrt((psi.I1-psi.m1)*(psi.I1+psi.m1+1))
    ket = BasisState(psi.J, psi.mJ, psi.I1, psi.m1+1, psi.I2, psi.m2)
    return State([(amp,ket)])

def I1m(psi):
    amp = sqrt((psi.I1+psi.m1)*(psi.I1-psi.m1+1))
    ket = BasisState(psi.J, psi.mJ, psi.I1, psi.m1-1, psi.I2, psi.m2)
    return State([(amp,ket)])

def I2p(psi):
    amp = sqrt((psi.I2-psi.m2)*(psi.I2+psi.m2+1))
    ket = BasisState(psi.J, psi.mJ, psi.I1, psi.m1, psi.I2, psi.m2+1)
    return State([(amp,ket)])

def I2m(psi):
    amp = sqrt((psi.I2+psi.m2)*(psi.I2-psi.m2+1))
    ket = BasisState(psi.J, psi.mJ, psi.I1, psi.m1, psi.I2, psi.m2-1)
    return State([(amp,ket)])

# In terms of the above-defined ladder operators, we can write
# 
# $$J_x=\frac{1}{2}(J_++J_-);\quad
# J_y=\frac{1}{2i}(J_+-J_-),$$
# 
# and similarly for $I_{1x}, I_{1y}$ and $I_{2x}, I_{2y}$.

def Jx(psi):
    return .5*( Jp(psi) + Jm(psi) )

def Jy(psi):
    return -.5j*( Jp(psi) - Jm(psi) )

def I1x(psi):
    return .5*( I1p(psi) + I1m(psi) )

def I1y(psi):
    return -.5j*( I1p(psi) - I1m(psi) )

def I2x(psi):
    return .5*( I2p(psi) + I2m(psi) )

def I2y(psi):
    return -.5j*( I2p(psi) - I2m(psi) )

# ### Composition of operators
# 
# All operators defined above can only accept `BasisStates` as their inputs, and
# they all return `States` as output. To allow composition of operators,
# 
# $$\hat A\hat B|\psi\rangle=\hat A(\hat B(|\psi\rangle)),$$
# 
# define the following function.

def com(A, B, psi):
    ABpsi = State()
    # operate with A on all components in B|psi>
    for amp,cpt in B(psi):
        ABpsi += amp * A(cpt)
    return ABpsi

# ### Rotational term
# 
# The simplest term in the Hamiltonian simply gives the rotational levels:
# 
# $$H_\text{rot}=B_\text{rot}\vec J^2.$$

def Hrot(psi):
    return Brot * J2(psi)

# ### Terms with ang. momentum dot products
# 
# Note that the dot product of two angular momentum operators can be written in terms of the ladder operators as
# 
# $$\vec A\cdot\vec B=A_zB_z+\frac{1}{2}(A_+B_-+A_-B_+).$$
# 
# We have the following terms (from Table 1 of Ramsey's paper):
# 
# $$
# H_\text{c1}=c_1\vec I_1\cdot\vec J;\quad
# H_\text{c2}=c_2\vec I_2\cdot\vec J;\quad
# H_\text{c4}=c_4\vec I_1\cdot\vec I_2\\
# H_\text{c3a}=15c_3\frac{(\vec I_1\cdot\vec J)(\vec I_2\cdot\vec J)}{(2J+3)(2J-1)}
# =\frac{15c_3}{c_1c_2}\frac{H_\text{c1}H_\text{c2}}{(2J+3)(2J-1)}\\
# H_\text{c3b}=15c_3\frac{(\vec I_2\cdot\vec J)(\vec I_1\cdot\vec J)}{(2J+3)(2J-1)}
# =\frac{15c_3}{c_1c_2}\frac{H_\text{c2}H_\text{c1}}{(2J+3)(2J-1)}\\
# H_\text{c3c}=-10c_3\frac{(\vec I_1\cdot\vec I_2)\vec J^2}{(2J+3)(2J-1)}
# =\frac{-10c_3}{c_4 B_\text{rot}}\frac{H_\text{c4}H_\text{rot}}{(2J+3)(2J-1)}
# $$

def Hc1(psi):
    return c1 * ( com(I1z,Jz,psi) + .5*(com(I1p,Jm,psi)+com(I1m,Jp,psi)) )

def Hc2(psi):
    return c2 * ( com(I2z,Jz,psi) + .5*(com(I2p,Jm,psi)+com(I2m,Jp,psi)) )

def Hc4(psi):
    return c4 * ( com(I1z,I2z,psi) + .5*(com(I1p,I2m,psi)+com(I1m,I2p,psi)) )

def Hc3a(psi):
    return 15*c3/c1/c2 * com(Hc1,Hc2,psi) / ((2*psi.J+3)*(2*psi.J-1))

def Hc3b(psi):
    return 15*c3/c1/c2 * com(Hc2,Hc1,psi) / ((2*psi.J+3)*(2*psi.J-1))

def Hc3c(psi):
    return -10*c3/c4/Brot * com(Hc4,Hrot,psi) / ((2*psi.J+3)*(2*psi.J-1))

# The overall field-free Hamiltonian is

def Hff(psi):
    return Hrot(psi) + Hc1(psi) + Hc2(psi) + Hc3a(psi) + Hc3b(psi) \
            + Hc3c(psi) + Hc4(psi)

# ### Zeeman Hamiltonian
# 
# In order to separate the task of finding the matrix elements and the
# eigenvalues, the Hamiltonian
# 
# $$H^\text{Z}=-\frac{\mu_J}{J}(\vec J\cdot\vec B)-\frac{\mu_1}{I_1}(\vec
# I_1\cdot\vec B)-\frac{\mu_2}{I_2}(\vec I_2\cdot\vec B)$$
# 
# is best split into three matrices:
# 
# $$H^\text{Z}=B_xH^\text{Z}_x+B_yH^\text{Z}_y+B_zH^\text{Z}_z,$$
# 
# where
# 
# $$ H^\text{Z}_x = -\frac{\mu_J}{J}J_x -\frac{\mu_1}{I_1}I_{1x} -\frac{\mu_2}{I_2}I_{2x} $$
# $$ H^\text{Z}_y = -\frac{\mu_J}{J}J_y -\frac{\mu_1}{I_1}I_{1y} -\frac{\mu_2}{I_2}I_{2y} $$
# $$ H^\text{Z}_z = -\frac{\mu_J}{J}J_z -\frac{\mu_1}{I_1}I_{1z} -\frac{\mu_2}{I_2}I_{2z} $$
# 
# Note that we are using the convention $\mu_1=\mu_\text{Tl}$ and
# $\mu_2=\mu_\text{F}$. The terms involving division by $J$ are only valid for
# states with $J\ne0$ (of course!).

def HZx(psi):
    if psi.J != 0:
        return -mu_J/psi.J*Jx(psi) - mu_Tl/psi.I1*I1x(psi) - mu_F/psi.I2*I2x(psi)
    else:
        return -mu_Tl/psi.I1*I1x(psi) - mu_F/psi.I2*I2x(psi)

def HZy(psi):
    if psi.J != 0:
        return -mu_J/psi.J*Jy(psi) - mu_Tl/psi.I1*I1y(psi) - mu_F/psi.I2*I2y(psi)
    else:
        return -mu_Tl/psi.I1*I1y(psi) - mu_F/psi.I2*I2y(psi)
    
def HZz(psi):
    if psi.J != 0:
        return -mu_J/psi.J*Jz(psi) - mu_Tl/psi.I1*I1z(psi) - mu_F/psi.I2*I2z(psi)
    else:
        return -mu_Tl/psi.I1*I1z(psi) - mu_F/psi.I2*I2z(psi)

# ### Stark Hamiltonian
# 
# Again splitting the Hamiltonian into the three spatial components, we have
# 
# $$H^\text{S}=-\vec d\cdot\vec E
# =E_xH^\text{S}_x+E_yH^\text{S}_y+E_zH^\text{S}_z.$$
# 
# To find the effect of the electric dipole operators (written in terms of the
# spherical harmonics)
# 
# $$\vec d
# =d_\text{TlF}\begin{pmatrix}\hat d_x\\\hat d_y\\\hat d_z\end{pmatrix}
# =d_\text{TlF}\begin{pmatrix}\sin\theta\cos\phi\\\sin\theta\sin\phi\\\cos\theta\end{pmatrix}
# =d_\text{TlF}\sqrt{\frac{2\pi}{3}}\begin{pmatrix}
# Y_1^{-1}-Y_1^1\\
# i(Y_1^{-1}+Y_1^1)\\
# \sqrt2Y_1^0\end{pmatrix}$$
# 
# on the eigenstates $|J,m,\ldots\rangle$, we need to find their matrix elements.
# The wavefunctions are $\langle\theta,\phi|J,m\rangle=Y_J^m$, so the matrix
# elements of the spherical harmonics are
# 
# $$
# \langle J',m'|Y_1^M||J,m\rangle
# =\int(Y_{J'}^{m'})^*Y_1^MY_J^md\Omega
# =(-1)^{m}\int(Y_{J'}^{m'})^*(Y_1^{-M})^*Y_J^md\Omega.
# $$
# 
# According to
# [Wikipedia](https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients#Relation_to_spherical_harmonics),
# this evaluates to
# 
# $$
# \sqrt{\frac{2\pi}{3}}
# \langle J',m'|Y_1^M||J,m\rangle
# =(-1)^{M}
#  \sqrt{\frac{(2 J' + 1)}{2(2 J + 1)}}
#     \langle J' \, 0 \, 1 \, 0 | J \, 0 \rangle
#     \langle J' \, m' \, 1 \, -M | J \, m \rangle
# $$
# 
# This can be partially evaluated using the following Mathematica function:
# 
# ```mathematica
# coeffs[M_] := Table[(-1)^M Sqrt[(2 Jp + 1)/(2 (2 J + 1))]
#      ClebschGordan[{Jp, mp}, {1, -M}, {J, m}]
#      ClebschGordan[{Jp, 0}, {1, 0}, {J, 0}] // FullSimplify,
#    {mp, {m - 1, m, m + 1}}, {Jp, {J - 1, J + 1}}
#    ] // MatrixForm
# ```
# 
# The result for $M=0$ is nonzero for $m'=m$:
# 
# $$
# \begin{aligned}
# \sqrt{\frac{(J-m)(J+m)}{8J^2-2}}&\quad\text{for $J'=J-1$}\\
# \sqrt{\frac{(J-m+1)(J+m+1)}{6+8J(J+2)}}&\quad\text{for $J'=J+1$}
# \end{aligned}
# $$
# 
# For $M=-1$, we need $m'=m-1$:
# 
# $$
# \begin{aligned}
# -\frac{1}{2}
# \sqrt{\frac{(J+m)(J-1+m)}{4J^2-1}}&\quad\text{for $J'=J-1$}\\
# \frac{1}{2}
# \sqrt{\frac{(J+1-m)(J+2-m)}{3+4J(J+2)}}&\quad\text{for $J'=J+1$}
# \end{aligned}
# $$
# 
# For $M=1$, we need $m'=m+1$:
# 
# $$
# \begin{aligned}
# -\frac{1}{2}
# \sqrt{\frac{(J-m)(J-1-m)}{4J^2-1}}&\quad\text{for $J'=J-1$}\\
# \frac{1}{2}
# \sqrt{\frac{(J+1+m)(J+2+m)}{3+4J(J+2)}}&\quad\text{for $J'=J+1$}
# \end{aligned}
# $$
# 
# These three cases can be written in Python as the operators:

def R10(psi):
    amp1 = sqrt((psi.J-psi.mJ)*(psi.J+psi.mJ)/(8*psi.J**2-2))
    ket1 = BasisState(psi.J-1, psi.mJ, psi.I1, psi.m1, psi.I2, psi.m2)
    amp2 = sqrt((psi.J-psi.mJ+1)*(psi.J+psi.mJ+1)/(6+8*psi.J*(psi.J+2)))
    ket2 = BasisState(psi.J+1, psi.mJ, psi.I1, psi.m1, psi.I2, psi.m2)
    return State([(amp1,ket1),(amp2,ket2)])

def R1m(psi):
    amp1 = -.5*sqrt((psi.J+psi.mJ)*(psi.J+psi.mJ-1)/(4*psi.J**2-1))
    ket1 = BasisState(psi.J-1, psi.mJ-1, psi.I1, psi.m1, psi.I2, psi.m2)
    amp2 = .5*sqrt((psi.J-psi.mJ+1)*(psi.J-psi.mJ+2)/(3+4*psi.J*(psi.J+2)))
    ket2 = BasisState(psi.J+1, psi.mJ-1, psi.I1, psi.m1, psi.I2, psi.m2)
    return State([(amp1,ket1),(amp2,ket2)])

def R1p(psi):
    amp1 = -.5*sqrt((psi.J-psi.mJ)*(psi.J-psi.mJ-1)/(4*psi.J**2-1))
    ket1 = BasisState(psi.J-1, psi.mJ+1, psi.I1, psi.m1, psi.I2, psi.m2)
    amp2 = .5*sqrt((psi.J+psi.mJ+1)*(psi.J+psi.mJ+2)/(3+4*psi.J*(psi.J+2)))
    ket2 = BasisState(psi.J+1, psi.mJ+1, psi.I1, psi.m1, psi.I2, psi.m2)
    return State([(amp1,ket1),(amp2,ket2)])

# In terms of the operators
# 
# $$
# R^M_1\equiv\sqrt{\frac{2\pi}{3}}Y_1^M
# $$
# 
# and the molecular dipole moment $d_\text{TlF}$, the three Stark Hamiltonians are
# 
# $$
# \begin{align}
# H^\text{S}_x&=-d_\text{TlF}(R^{-1}_1-R^1_1)\\
# H^\text{S}_y&=-d_\text{TlF}i(R^{-1}_1+R^1_1)\\
# H^\text{S}_z&=-d_\text{TlF}\sqrt2R^0_1
# \end{align}
# $$
# 
# In Python:

def HSx(psi):
    return -D_TlF * ( R1m(psi) - R1p(psi) )

def HSy(psi):
    return -D_TlF * 1j * ( R1m(psi) + R1p(psi) )

def HSz(psi):
    return -D_TlF * sqrt(2)*R10(psi)

####################################
### Finding the matrix elements
####################################

# With all the operators defined, we can evaluate the matrix elements for a given
# range of quantum numbers. Write down the basis as a list of `BasisState`
# components:

QN = np.array([BasisState(J,mJ,I_Tl,m1,I_F,m2)
      for J in range(Jmax+1)
      for mJ in range(-J,J+1)
      for m1 in np.arange(-I_Tl,I_Tl+1)
      for m2 in np.arange(-I_F,I_F+1)])

# The field-free and Stark/Zeeman components of the Hamiltonian then have the
# matrix elements

def HMatElems(H, QN):
    result = np.empty((len(QN),len(QN)), dtype=complex)
    for i,a in enumerate(QN):
        for j,b in enumerate(QN):
            result[i,j] = (1*a)@H(b)
    return result

# Now find the numerical values of all the matrix elements

if __name__ == "__main__":
    Hff_m = HMatElems(Hff, QN)
    HSx_m = HMatElems(HSx, QN)
    HSy_m = HMatElems(HSy, QN)
    HSz_m = HMatElems(HSz, QN)
    HZx_m = HMatElems(HZx, QN)
    HZy_m = HMatElems(HZy, QN)
    HZz_m = HMatElems(HZz, QN)
