??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
DepthToSpace

input"T
output"T"	
Ttype"

block_sizeint(0":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
.
Identity

input"T
output"T"	
Ttype
;
Maximum
x"T
y"T
z"T"
Ttype:

2	?
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
;
Minimum
x"T
y"T
z"T"
Ttype:

2	?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
H
ShardedFilename
basename	
shard

num_shards
filename
1
Square
x"T
y"T"
Ttype:

2	
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.11.02
b'unknown'8??

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
dtype0	*
_output_shapes
: *
_class
loc:@global_step*
shape: 
?
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
?
input_tensorPlaceholder*6
shape-:+???????????????????????????*
dtype0*A
_output_shapes/
-:+???????????????????????????
Z
ConstConst*!
valueB"??>???>??>*
dtype0*
_output_shapes
:
k
subSubinput_tensorConst*
T0*A
_output_shapes/
-:+???????????????????????????
?
-skip/conv2d_weight_norm/wn_g/Initializer/onesConst*
valueB*  ??*/
_class%
#!loc:@skip/conv2d_weight_norm/wn_g*
dtype0*
_output_shapes
:
?
skip/conv2d_weight_norm/wn_g
VariableV2*/
_class%
#!loc:@skip/conv2d_weight_norm/wn_g*
shape:*
dtype0*
_output_shapes
:
?
#skip/conv2d_weight_norm/wn_g/AssignAssignskip/conv2d_weight_norm/wn_g-skip/conv2d_weight_norm/wn_g/Initializer/ones*
T0*/
_class%
#!loc:@skip/conv2d_weight_norm/wn_g*
_output_shapes
:
?
!skip/conv2d_weight_norm/wn_g/readIdentityskip/conv2d_weight_norm/wn_g*
T0*/
_class%
#!loc:@skip/conv2d_weight_norm/wn_g*
_output_shapes
:
?
?skip/conv2d_weight_norm/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *1
_class'
%#loc:@skip/conv2d_weight_norm/kernel*
dtype0*
_output_shapes
:
?
=skip/conv2d_weight_norm/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *???*1
_class'
%#loc:@skip/conv2d_weight_norm/kernel
?
=skip/conv2d_weight_norm/kernel/Initializer/random_uniform/maxConst*
valueB
 *??>*1
_class'
%#loc:@skip/conv2d_weight_norm/kernel*
dtype0*
_output_shapes
: 
?
Gskip/conv2d_weight_norm/kernel/Initializer/random_uniform/RandomUniformRandomUniform?skip/conv2d_weight_norm/kernel/Initializer/random_uniform/shape*
T0*1
_class'
%#loc:@skip/conv2d_weight_norm/kernel*
dtype0*&
_output_shapes
:
?
=skip/conv2d_weight_norm/kernel/Initializer/random_uniform/subSub=skip/conv2d_weight_norm/kernel/Initializer/random_uniform/max=skip/conv2d_weight_norm/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@skip/conv2d_weight_norm/kernel*
_output_shapes
: 
?
=skip/conv2d_weight_norm/kernel/Initializer/random_uniform/mulMulGskip/conv2d_weight_norm/kernel/Initializer/random_uniform/RandomUniform=skip/conv2d_weight_norm/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@skip/conv2d_weight_norm/kernel*&
_output_shapes
:
?
9skip/conv2d_weight_norm/kernel/Initializer/random_uniformAdd=skip/conv2d_weight_norm/kernel/Initializer/random_uniform/mul=skip/conv2d_weight_norm/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@skip/conv2d_weight_norm/kernel*&
_output_shapes
:
?
skip/conv2d_weight_norm/kernel
VariableV2*1
_class'
%#loc:@skip/conv2d_weight_norm/kernel*
shape:*
dtype0*&
_output_shapes
:
?
%skip/conv2d_weight_norm/kernel/AssignAssignskip/conv2d_weight_norm/kernel9skip/conv2d_weight_norm/kernel/Initializer/random_uniform*
T0*1
_class'
%#loc:@skip/conv2d_weight_norm/kernel*&
_output_shapes
:
?
#skip/conv2d_weight_norm/kernel/readIdentityskip/conv2d_weight_norm/kernel*
T0*1
_class'
%#loc:@skip/conv2d_weight_norm/kernel*&
_output_shapes
:
?
.skip/conv2d_weight_norm/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    */
_class%
#!loc:@skip/conv2d_weight_norm/bias
?
skip/conv2d_weight_norm/bias
VariableV2*
dtype0*
_output_shapes
:*/
_class%
#!loc:@skip/conv2d_weight_norm/bias*
shape:
?
#skip/conv2d_weight_norm/bias/AssignAssignskip/conv2d_weight_norm/bias.skip/conv2d_weight_norm/bias/Initializer/zeros*
_output_shapes
:*
T0*/
_class%
#!loc:@skip/conv2d_weight_norm/bias
?
!skip/conv2d_weight_norm/bias/readIdentityskip/conv2d_weight_norm/bias*
T0*/
_class%
#!loc:@skip/conv2d_weight_norm/bias*
_output_shapes
:
v
%skip/conv2d_weight_norm/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
~
skip/conv2d_weight_norm/SquareSquare#skip/conv2d_weight_norm/kernel/read*&
_output_shapes
:*
T0
?
-skip/conv2d_weight_norm/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
?
skip/conv2d_weight_norm/SumSumskip/conv2d_weight_norm/Square-skip/conv2d_weight_norm/Sum/reduction_indices*
T0*
_output_shapes
:
h
skip/conv2d_weight_norm/RsqrtRsqrtskip/conv2d_weight_norm/Sum*
T0*
_output_shapes
:
?
skip/conv2d_weight_norm/mulMulskip/conv2d_weight_norm/Rsqrt!skip/conv2d_weight_norm/wn_g/read*
T0*
_output_shapes
:
?
skip/conv2d_weight_norm/mul_1Mul#skip/conv2d_weight_norm/kernel/readskip/conv2d_weight_norm/mul*
T0*&
_output_shapes
:
?
skip/conv2d_weight_norm/Conv2DConv2Dsubskip/conv2d_weight_norm/mul_1*
paddingSAME*A
_output_shapes/
-:+???????????????????????????*
T0*
strides

?
skip/conv2d_weight_norm/BiasAddBiasAddskip/conv2d_weight_norm/Conv2D!skip/conv2d_weight_norm/bias/read*
T0*A
_output_shapes/
-:+???????????????????????????
?
skip/DepthToSpaceDepthToSpaceskip/conv2d_weight_norm/BiasAdd*
T0*A
_output_shapes/
-:+???????????????????????????*

block_size
?
.input/conv2d_weight_norm/wn_g/Initializer/onesConst*
valueB *  ??*0
_class&
$"loc:@input/conv2d_weight_norm/wn_g*
dtype0*
_output_shapes
: 
?
input/conv2d_weight_norm/wn_g
VariableV2*
dtype0*
_output_shapes
: *0
_class&
$"loc:@input/conv2d_weight_norm/wn_g*
shape: 
?
$input/conv2d_weight_norm/wn_g/AssignAssigninput/conv2d_weight_norm/wn_g.input/conv2d_weight_norm/wn_g/Initializer/ones*
_output_shapes
: *
T0*0
_class&
$"loc:@input/conv2d_weight_norm/wn_g
?
"input/conv2d_weight_norm/wn_g/readIdentityinput/conv2d_weight_norm/wn_g*
T0*0
_class&
$"loc:@input/conv2d_weight_norm/wn_g*
_output_shapes
: 
?
@input/conv2d_weight_norm/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"             *2
_class(
&$loc:@input/conv2d_weight_norm/kernel
?
>input/conv2d_weight_norm/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *OS?*2
_class(
&$loc:@input/conv2d_weight_norm/kernel
?
>input/conv2d_weight_norm/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *OS>*2
_class(
&$loc:@input/conv2d_weight_norm/kernel
?
Hinput/conv2d_weight_norm/kernel/Initializer/random_uniform/RandomUniformRandomUniform@input/conv2d_weight_norm/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *
T0*2
_class(
&$loc:@input/conv2d_weight_norm/kernel
?
>input/conv2d_weight_norm/kernel/Initializer/random_uniform/subSub>input/conv2d_weight_norm/kernel/Initializer/random_uniform/max>input/conv2d_weight_norm/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*2
_class(
&$loc:@input/conv2d_weight_norm/kernel
?
>input/conv2d_weight_norm/kernel/Initializer/random_uniform/mulMulHinput/conv2d_weight_norm/kernel/Initializer/random_uniform/RandomUniform>input/conv2d_weight_norm/kernel/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@input/conv2d_weight_norm/kernel*&
_output_shapes
: 
?
:input/conv2d_weight_norm/kernel/Initializer/random_uniformAdd>input/conv2d_weight_norm/kernel/Initializer/random_uniform/mul>input/conv2d_weight_norm/kernel/Initializer/random_uniform/min*&
_output_shapes
: *
T0*2
_class(
&$loc:@input/conv2d_weight_norm/kernel
?
input/conv2d_weight_norm/kernel
VariableV2*
shape: *
dtype0*&
_output_shapes
: *2
_class(
&$loc:@input/conv2d_weight_norm/kernel
?
&input/conv2d_weight_norm/kernel/AssignAssigninput/conv2d_weight_norm/kernel:input/conv2d_weight_norm/kernel/Initializer/random_uniform*
T0*2
_class(
&$loc:@input/conv2d_weight_norm/kernel*&
_output_shapes
: 
?
$input/conv2d_weight_norm/kernel/readIdentityinput/conv2d_weight_norm/kernel*
T0*2
_class(
&$loc:@input/conv2d_weight_norm/kernel*&
_output_shapes
: 
?
/input/conv2d_weight_norm/bias/Initializer/zerosConst*
valueB *    *0
_class&
$"loc:@input/conv2d_weight_norm/bias*
dtype0*
_output_shapes
: 
?
input/conv2d_weight_norm/bias
VariableV2*
dtype0*
_output_shapes
: *0
_class&
$"loc:@input/conv2d_weight_norm/bias*
shape: 
?
$input/conv2d_weight_norm/bias/AssignAssigninput/conv2d_weight_norm/bias/input/conv2d_weight_norm/bias/Initializer/zeros*
T0*0
_class&
$"loc:@input/conv2d_weight_norm/bias*
_output_shapes
: 
?
"input/conv2d_weight_norm/bias/readIdentityinput/conv2d_weight_norm/bias*
T0*0
_class&
$"loc:@input/conv2d_weight_norm/bias*
_output_shapes
: 
w
&input/conv2d_weight_norm/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
input/conv2d_weight_norm/SquareSquare$input/conv2d_weight_norm/kernel/read*
T0*&
_output_shapes
: 
?
.input/conv2d_weight_norm/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
?
input/conv2d_weight_norm/SumSuminput/conv2d_weight_norm/Square.input/conv2d_weight_norm/Sum/reduction_indices*
T0*
_output_shapes
: 
j
input/conv2d_weight_norm/RsqrtRsqrtinput/conv2d_weight_norm/Sum*
T0*
_output_shapes
: 
?
input/conv2d_weight_norm/mulMulinput/conv2d_weight_norm/Rsqrt"input/conv2d_weight_norm/wn_g/read*
T0*
_output_shapes
: 
?
input/conv2d_weight_norm/mul_1Mul$input/conv2d_weight_norm/kernel/readinput/conv2d_weight_norm/mul*
T0*&
_output_shapes
: 
?
input/conv2d_weight_norm/Conv2DConv2Dsubinput/conv2d_weight_norm/mul_1*
strides
*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? *
T0
?
 input/conv2d_weight_norm/BiasAddBiasAddinput/conv2d_weight_norm/Conv2D"input/conv2d_weight_norm/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
"layer0/conv0/wn_g/Initializer/onesConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??*$
_class
loc:@layer0/conv0/wn_g
?
layer0/conv0/wn_g
VariableV2*$
_class
loc:@layer0/conv0/wn_g*
shape:?*
dtype0*
_output_shapes	
:?
?
layer0/conv0/wn_g/AssignAssignlayer0/conv0/wn_g"layer0/conv0/wn_g/Initializer/ones*
T0*$
_class
loc:@layer0/conv0/wn_g*
_output_shapes	
:?
?
layer0/conv0/wn_g/readIdentitylayer0/conv0/wn_g*
T0*$
_class
loc:@layer0/conv0/wn_g*
_output_shapes	
:?
?
4layer0/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *&
_class
loc:@layer0/conv0/kernel*
dtype0*
_output_shapes
:
?
2layer0/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer0/conv0/kernel*
dtype0*
_output_shapes
: 
?
2layer0/conv0/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *?2?=*&
_class
loc:@layer0/conv0/kernel
?
<layer0/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer0/conv0/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
: ?*
T0*&
_class
loc:@layer0/conv0/kernel
?
2layer0/conv0/kernel/Initializer/random_uniform/subSub2layer0/conv0/kernel/Initializer/random_uniform/max2layer0/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer0/conv0/kernel*
_output_shapes
: 
?
2layer0/conv0/kernel/Initializer/random_uniform/mulMul<layer0/conv0/kernel/Initializer/random_uniform/RandomUniform2layer0/conv0/kernel/Initializer/random_uniform/sub*'
_output_shapes
: ?*
T0*&
_class
loc:@layer0/conv0/kernel
?
.layer0/conv0/kernel/Initializer/random_uniformAdd2layer0/conv0/kernel/Initializer/random_uniform/mul2layer0/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer0/conv0/kernel*'
_output_shapes
: ?
?
layer0/conv0/kernel
VariableV2*
dtype0*'
_output_shapes
: ?*&
_class
loc:@layer0/conv0/kernel*
shape: ?
?
layer0/conv0/kernel/AssignAssignlayer0/conv0/kernel.layer0/conv0/kernel/Initializer/random_uniform*'
_output_shapes
: ?*
T0*&
_class
loc:@layer0/conv0/kernel
?
layer0/conv0/kernel/readIdentitylayer0/conv0/kernel*
T0*&
_class
loc:@layer0/conv0/kernel*'
_output_shapes
: ?
?
#layer0/conv0/bias/Initializer/zerosConst*
valueB?*    *$
_class
loc:@layer0/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer0/conv0/bias
VariableV2*$
_class
loc:@layer0/conv0/bias*
shape:?*
dtype0*
_output_shapes	
:?
?
layer0/conv0/bias/AssignAssignlayer0/conv0/bias#layer0/conv0/bias/Initializer/zeros*
_output_shapes	
:?*
T0*$
_class
loc:@layer0/conv0/bias
?
layer0/conv0/bias/readIdentitylayer0/conv0/bias*
T0*$
_class
loc:@layer0/conv0/bias*
_output_shapes	
:?
k
layer0/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer0/conv0/SquareSquarelayer0/conv0/kernel/read*
T0*'
_output_shapes
: ?
w
"layer0/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
v
layer0/conv0/SumSumlayer0/conv0/Square"layer0/conv0/Sum/reduction_indices*
_output_shapes	
:?*
T0
S
layer0/conv0/RsqrtRsqrtlayer0/conv0/Sum*
T0*
_output_shapes	
:?
i
layer0/conv0/mulMullayer0/conv0/Rsqrtlayer0/conv0/wn_g/read*
T0*
_output_shapes	
:?
w
layer0/conv0/mul_1Mullayer0/conv0/kernel/readlayer0/conv0/mul*
T0*'
_output_shapes
: ?
?
layer0/conv0/Conv2DConv2D input/conv2d_weight_norm/BiasAddlayer0/conv0/mul_1*
paddingSAME*B
_output_shapes0
.:,????????????????????????????*
T0*
strides

?
layer0/conv0/BiasAddBiasAddlayer0/conv0/Conv2Dlayer0/conv0/bias/read*
T0*B
_output_shapes0
.:,????????????????????????????
v
layer0/ReluRelulayer0/conv0/BiasAdd*B
_output_shapes0
.:,????????????????????????????*
T0
?
"layer0/conv1/wn_g/Initializer/onesConst*
dtype0*
_output_shapes
: *
valueB *  ??*$
_class
loc:@layer0/conv1/wn_g

layer0/conv1/wn_g
VariableV2*$
_class
loc:@layer0/conv1/wn_g*
shape: *
dtype0*
_output_shapes
: 
?
layer0/conv1/wn_g/AssignAssignlayer0/conv1/wn_g"layer0/conv1/wn_g/Initializer/ones*
T0*$
_class
loc:@layer0/conv1/wn_g*
_output_shapes
: 
?
layer0/conv1/wn_g/readIdentitylayer0/conv1/wn_g*
T0*$
_class
loc:@layer0/conv1/wn_g*
_output_shapes
: 
?
4layer0/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *&
_class
loc:@layer0/conv1/kernel*
dtype0*
_output_shapes
:
?
2layer0/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer0/conv1/kernel*
dtype0*
_output_shapes
: 
?
2layer0/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer0/conv1/kernel*
dtype0*
_output_shapes
: 
?
<layer0/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer0/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*&
_class
loc:@layer0/conv1/kernel
?
2layer0/conv1/kernel/Initializer/random_uniform/subSub2layer0/conv1/kernel/Initializer/random_uniform/max2layer0/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer0/conv1/kernel*
_output_shapes
: 
?
2layer0/conv1/kernel/Initializer/random_uniform/mulMul<layer0/conv1/kernel/Initializer/random_uniform/RandomUniform2layer0/conv1/kernel/Initializer/random_uniform/sub*'
_output_shapes
:? *
T0*&
_class
loc:@layer0/conv1/kernel
?
.layer0/conv1/kernel/Initializer/random_uniformAdd2layer0/conv1/kernel/Initializer/random_uniform/mul2layer0/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer0/conv1/kernel*'
_output_shapes
:? 
?
layer0/conv1/kernel
VariableV2*&
_class
loc:@layer0/conv1/kernel*
shape:? *
dtype0*'
_output_shapes
:? 
?
layer0/conv1/kernel/AssignAssignlayer0/conv1/kernel.layer0/conv1/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer0/conv1/kernel*'
_output_shapes
:? 
?
layer0/conv1/kernel/readIdentitylayer0/conv1/kernel*
T0*&
_class
loc:@layer0/conv1/kernel*'
_output_shapes
:? 
?
#layer0/conv1/bias/Initializer/zerosConst*
valueB *    *$
_class
loc:@layer0/conv1/bias*
dtype0*
_output_shapes
: 

layer0/conv1/bias
VariableV2*$
_class
loc:@layer0/conv1/bias*
shape: *
dtype0*
_output_shapes
: 
?
layer0/conv1/bias/AssignAssignlayer0/conv1/bias#layer0/conv1/bias/Initializer/zeros*
T0*$
_class
loc:@layer0/conv1/bias*
_output_shapes
: 
?
layer0/conv1/bias/readIdentitylayer0/conv1/bias*
T0*$
_class
loc:@layer0/conv1/bias*
_output_shapes
: 
k
layer0/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer0/conv1/SquareSquarelayer0/conv1/kernel/read*'
_output_shapes
:? *
T0
w
"layer0/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
u
layer0/conv1/SumSumlayer0/conv1/Square"layer0/conv1/Sum/reduction_indices*
T0*
_output_shapes
: 
R
layer0/conv1/RsqrtRsqrtlayer0/conv1/Sum*
T0*
_output_shapes
: 
h
layer0/conv1/mulMullayer0/conv1/Rsqrtlayer0/conv1/wn_g/read*
T0*
_output_shapes
: 
w
layer0/conv1/mul_1Mullayer0/conv1/kernel/readlayer0/conv1/mul*
T0*'
_output_shapes
:? 
?
layer0/conv1/Conv2DConv2Dlayer0/Relulayer0/conv1/mul_1*
strides
*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? *
T0
?
layer0/conv1/BiasAddBiasAddlayer0/conv1/Conv2Dlayer0/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 
?

layer0/addAddlayer0/conv1/BiasAdd input/conv2d_weight_norm/BiasAdd*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
"layer1/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*$
_class
loc:@layer1/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer1/conv0/wn_g
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer1/conv0/wn_g
?
layer1/conv0/wn_g/AssignAssignlayer1/conv0/wn_g"layer1/conv0/wn_g/Initializer/ones*
_output_shapes	
:?*
T0*$
_class
loc:@layer1/conv0/wn_g
?
layer1/conv0/wn_g/readIdentitylayer1/conv0/wn_g*
_output_shapes	
:?*
T0*$
_class
loc:@layer1/conv0/wn_g
?
4layer1/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *&
_class
loc:@layer1/conv0/kernel*
dtype0*
_output_shapes
:
?
2layer1/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer1/conv0/kernel*
dtype0*
_output_shapes
: 
?
2layer1/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer1/conv0/kernel*
dtype0*
_output_shapes
: 
?
<layer1/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer1/conv0/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
: ?*
T0*&
_class
loc:@layer1/conv0/kernel
?
2layer1/conv0/kernel/Initializer/random_uniform/subSub2layer1/conv0/kernel/Initializer/random_uniform/max2layer1/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer1/conv0/kernel*
_output_shapes
: 
?
2layer1/conv0/kernel/Initializer/random_uniform/mulMul<layer1/conv0/kernel/Initializer/random_uniform/RandomUniform2layer1/conv0/kernel/Initializer/random_uniform/sub*'
_output_shapes
: ?*
T0*&
_class
loc:@layer1/conv0/kernel
?
.layer1/conv0/kernel/Initializer/random_uniformAdd2layer1/conv0/kernel/Initializer/random_uniform/mul2layer1/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer1/conv0/kernel*'
_output_shapes
: ?
?
layer1/conv0/kernel
VariableV2*&
_class
loc:@layer1/conv0/kernel*
shape: ?*
dtype0*'
_output_shapes
: ?
?
layer1/conv0/kernel/AssignAssignlayer1/conv0/kernel.layer1/conv0/kernel/Initializer/random_uniform*'
_output_shapes
: ?*
T0*&
_class
loc:@layer1/conv0/kernel
?
layer1/conv0/kernel/readIdentitylayer1/conv0/kernel*
T0*&
_class
loc:@layer1/conv0/kernel*'
_output_shapes
: ?
?
#layer1/conv0/bias/Initializer/zerosConst*
valueB?*    *$
_class
loc:@layer1/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer1/conv0/bias
VariableV2*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer1/conv0/bias*
shape:?
?
layer1/conv0/bias/AssignAssignlayer1/conv0/bias#layer1/conv0/bias/Initializer/zeros*
T0*$
_class
loc:@layer1/conv0/bias*
_output_shapes	
:?
?
layer1/conv0/bias/readIdentitylayer1/conv0/bias*
T0*$
_class
loc:@layer1/conv0/bias*
_output_shapes	
:?
k
layer1/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer1/conv0/SquareSquarelayer1/conv0/kernel/read*
T0*'
_output_shapes
: ?
w
"layer1/conv0/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
v
layer1/conv0/SumSumlayer1/conv0/Square"layer1/conv0/Sum/reduction_indices*
T0*
_output_shapes	
:?
S
layer1/conv0/RsqrtRsqrtlayer1/conv0/Sum*
T0*
_output_shapes	
:?
i
layer1/conv0/mulMullayer1/conv0/Rsqrtlayer1/conv0/wn_g/read*
_output_shapes	
:?*
T0
w
layer1/conv0/mul_1Mullayer1/conv0/kernel/readlayer1/conv0/mul*
T0*'
_output_shapes
: ?
?
layer1/conv0/Conv2DConv2D
layer0/addlayer1/conv0/mul_1*B
_output_shapes0
.:,????????????????????????????*
T0*
strides
*
paddingSAME
?
layer1/conv0/BiasAddBiasAddlayer1/conv0/Conv2Dlayer1/conv0/bias/read*B
_output_shapes0
.:,????????????????????????????*
T0
v
layer1/ReluRelulayer1/conv0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
"layer1/conv1/wn_g/Initializer/onesConst*
valueB *  ??*$
_class
loc:@layer1/conv1/wn_g*
dtype0*
_output_shapes
: 

layer1/conv1/wn_g
VariableV2*
dtype0*
_output_shapes
: *$
_class
loc:@layer1/conv1/wn_g*
shape: 
?
layer1/conv1/wn_g/AssignAssignlayer1/conv1/wn_g"layer1/conv1/wn_g/Initializer/ones*
_output_shapes
: *
T0*$
_class
loc:@layer1/conv1/wn_g
?
layer1/conv1/wn_g/readIdentitylayer1/conv1/wn_g*
T0*$
_class
loc:@layer1/conv1/wn_g*
_output_shapes
: 
?
4layer1/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *&
_class
loc:@layer1/conv1/kernel*
dtype0*
_output_shapes
:
?
2layer1/conv1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?2??*&
_class
loc:@layer1/conv1/kernel
?
2layer1/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer1/conv1/kernel*
dtype0*
_output_shapes
: 
?
<layer1/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer1/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*&
_class
loc:@layer1/conv1/kernel
?
2layer1/conv1/kernel/Initializer/random_uniform/subSub2layer1/conv1/kernel/Initializer/random_uniform/max2layer1/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer1/conv1/kernel*
_output_shapes
: 
?
2layer1/conv1/kernel/Initializer/random_uniform/mulMul<layer1/conv1/kernel/Initializer/random_uniform/RandomUniform2layer1/conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer1/conv1/kernel*'
_output_shapes
:? 
?
.layer1/conv1/kernel/Initializer/random_uniformAdd2layer1/conv1/kernel/Initializer/random_uniform/mul2layer1/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer1/conv1/kernel*'
_output_shapes
:? 
?
layer1/conv1/kernel
VariableV2*&
_class
loc:@layer1/conv1/kernel*
shape:? *
dtype0*'
_output_shapes
:? 
?
layer1/conv1/kernel/AssignAssignlayer1/conv1/kernel.layer1/conv1/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer1/conv1/kernel*'
_output_shapes
:? 
?
layer1/conv1/kernel/readIdentitylayer1/conv1/kernel*'
_output_shapes
:? *
T0*&
_class
loc:@layer1/conv1/kernel
?
#layer1/conv1/bias/Initializer/zerosConst*
valueB *    *$
_class
loc:@layer1/conv1/bias*
dtype0*
_output_shapes
: 

layer1/conv1/bias
VariableV2*
shape: *
dtype0*
_output_shapes
: *$
_class
loc:@layer1/conv1/bias
?
layer1/conv1/bias/AssignAssignlayer1/conv1/bias#layer1/conv1/bias/Initializer/zeros*
_output_shapes
: *
T0*$
_class
loc:@layer1/conv1/bias
?
layer1/conv1/bias/readIdentitylayer1/conv1/bias*
T0*$
_class
loc:@layer1/conv1/bias*
_output_shapes
: 
k
layer1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer1/conv1/SquareSquarelayer1/conv1/kernel/read*
T0*'
_output_shapes
:? 
w
"layer1/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
u
layer1/conv1/SumSumlayer1/conv1/Square"layer1/conv1/Sum/reduction_indices*
_output_shapes
: *
T0
R
layer1/conv1/RsqrtRsqrtlayer1/conv1/Sum*
_output_shapes
: *
T0
h
layer1/conv1/mulMullayer1/conv1/Rsqrtlayer1/conv1/wn_g/read*
T0*
_output_shapes
: 
w
layer1/conv1/mul_1Mullayer1/conv1/kernel/readlayer1/conv1/mul*
T0*'
_output_shapes
:? 
?
layer1/conv1/Conv2DConv2Dlayer1/Relulayer1/conv1/mul_1*A
_output_shapes/
-:+??????????????????????????? *
T0*
strides
*
paddingSAME
?
layer1/conv1/BiasAddBiasAddlayer1/conv1/Conv2Dlayer1/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 


layer1/addAddlayer1/conv1/BiasAdd
layer0/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
"layer2/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*$
_class
loc:@layer2/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer2/conv0/wn_g
VariableV2*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer2/conv0/wn_g*
shape:?
?
layer2/conv0/wn_g/AssignAssignlayer2/conv0/wn_g"layer2/conv0/wn_g/Initializer/ones*
T0*$
_class
loc:@layer2/conv0/wn_g*
_output_shapes	
:?
?
layer2/conv0/wn_g/readIdentitylayer2/conv0/wn_g*
T0*$
_class
loc:@layer2/conv0/wn_g*
_output_shapes	
:?
?
4layer2/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *&
_class
loc:@layer2/conv0/kernel*
dtype0*
_output_shapes
:
?
2layer2/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer2/conv0/kernel*
dtype0*
_output_shapes
: 
?
2layer2/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer2/conv0/kernel*
dtype0*
_output_shapes
: 
?
<layer2/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer2/conv0/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@layer2/conv0/kernel*
dtype0*'
_output_shapes
: ?
?
2layer2/conv0/kernel/Initializer/random_uniform/subSub2layer2/conv0/kernel/Initializer/random_uniform/max2layer2/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer2/conv0/kernel*
_output_shapes
: 
?
2layer2/conv0/kernel/Initializer/random_uniform/mulMul<layer2/conv0/kernel/Initializer/random_uniform/RandomUniform2layer2/conv0/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer2/conv0/kernel*'
_output_shapes
: ?
?
.layer2/conv0/kernel/Initializer/random_uniformAdd2layer2/conv0/kernel/Initializer/random_uniform/mul2layer2/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer2/conv0/kernel*'
_output_shapes
: ?
?
layer2/conv0/kernel
VariableV2*
dtype0*'
_output_shapes
: ?*&
_class
loc:@layer2/conv0/kernel*
shape: ?
?
layer2/conv0/kernel/AssignAssignlayer2/conv0/kernel.layer2/conv0/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer2/conv0/kernel*'
_output_shapes
: ?
?
layer2/conv0/kernel/readIdentitylayer2/conv0/kernel*
T0*&
_class
loc:@layer2/conv0/kernel*'
_output_shapes
: ?
?
#layer2/conv0/bias/Initializer/zerosConst*
valueB?*    *$
_class
loc:@layer2/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer2/conv0/bias
VariableV2*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer2/conv0/bias*
shape:?
?
layer2/conv0/bias/AssignAssignlayer2/conv0/bias#layer2/conv0/bias/Initializer/zeros*
T0*$
_class
loc:@layer2/conv0/bias*
_output_shapes	
:?
?
layer2/conv0/bias/readIdentitylayer2/conv0/bias*
T0*$
_class
loc:@layer2/conv0/bias*
_output_shapes	
:?
k
layer2/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer2/conv0/SquareSquarelayer2/conv0/kernel/read*'
_output_shapes
: ?*
T0
w
"layer2/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
v
layer2/conv0/SumSumlayer2/conv0/Square"layer2/conv0/Sum/reduction_indices*
T0*
_output_shapes	
:?
S
layer2/conv0/RsqrtRsqrtlayer2/conv0/Sum*
T0*
_output_shapes	
:?
i
layer2/conv0/mulMullayer2/conv0/Rsqrtlayer2/conv0/wn_g/read*
T0*
_output_shapes	
:?
w
layer2/conv0/mul_1Mullayer2/conv0/kernel/readlayer2/conv0/mul*'
_output_shapes
: ?*
T0
?
layer2/conv0/Conv2DConv2D
layer1/addlayer2/conv0/mul_1*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,????????????????????????????
?
layer2/conv0/BiasAddBiasAddlayer2/conv0/Conv2Dlayer2/conv0/bias/read*
T0*B
_output_shapes0
.:,????????????????????????????
v
layer2/ReluRelulayer2/conv0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
"layer2/conv1/wn_g/Initializer/onesConst*
dtype0*
_output_shapes
: *
valueB *  ??*$
_class
loc:@layer2/conv1/wn_g

layer2/conv1/wn_g
VariableV2*
dtype0*
_output_shapes
: *$
_class
loc:@layer2/conv1/wn_g*
shape: 
?
layer2/conv1/wn_g/AssignAssignlayer2/conv1/wn_g"layer2/conv1/wn_g/Initializer/ones*
T0*$
_class
loc:@layer2/conv1/wn_g*
_output_shapes
: 
?
layer2/conv1/wn_g/readIdentitylayer2/conv1/wn_g*
T0*$
_class
loc:@layer2/conv1/wn_g*
_output_shapes
: 
?
4layer2/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *&
_class
loc:@layer2/conv1/kernel*
dtype0*
_output_shapes
:
?
2layer2/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer2/conv1/kernel*
dtype0*
_output_shapes
: 
?
2layer2/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer2/conv1/kernel*
dtype0*
_output_shapes
: 
?
<layer2/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer2/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*&
_class
loc:@layer2/conv1/kernel
?
2layer2/conv1/kernel/Initializer/random_uniform/subSub2layer2/conv1/kernel/Initializer/random_uniform/max2layer2/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer2/conv1/kernel*
_output_shapes
: 
?
2layer2/conv1/kernel/Initializer/random_uniform/mulMul<layer2/conv1/kernel/Initializer/random_uniform/RandomUniform2layer2/conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer2/conv1/kernel*'
_output_shapes
:? 
?
.layer2/conv1/kernel/Initializer/random_uniformAdd2layer2/conv1/kernel/Initializer/random_uniform/mul2layer2/conv1/kernel/Initializer/random_uniform/min*'
_output_shapes
:? *
T0*&
_class
loc:@layer2/conv1/kernel
?
layer2/conv1/kernel
VariableV2*&
_class
loc:@layer2/conv1/kernel*
shape:? *
dtype0*'
_output_shapes
:? 
?
layer2/conv1/kernel/AssignAssignlayer2/conv1/kernel.layer2/conv1/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer2/conv1/kernel*'
_output_shapes
:? 
?
layer2/conv1/kernel/readIdentitylayer2/conv1/kernel*
T0*&
_class
loc:@layer2/conv1/kernel*'
_output_shapes
:? 
?
#layer2/conv1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *$
_class
loc:@layer2/conv1/bias

layer2/conv1/bias
VariableV2*
dtype0*
_output_shapes
: *$
_class
loc:@layer2/conv1/bias*
shape: 
?
layer2/conv1/bias/AssignAssignlayer2/conv1/bias#layer2/conv1/bias/Initializer/zeros*
_output_shapes
: *
T0*$
_class
loc:@layer2/conv1/bias
?
layer2/conv1/bias/readIdentitylayer2/conv1/bias*
T0*$
_class
loc:@layer2/conv1/bias*
_output_shapes
: 
k
layer2/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
i
layer2/conv1/SquareSquarelayer2/conv1/kernel/read*
T0*'
_output_shapes
:? 
w
"layer2/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
u
layer2/conv1/SumSumlayer2/conv1/Square"layer2/conv1/Sum/reduction_indices*
T0*
_output_shapes
: 
R
layer2/conv1/RsqrtRsqrtlayer2/conv1/Sum*
T0*
_output_shapes
: 
h
layer2/conv1/mulMullayer2/conv1/Rsqrtlayer2/conv1/wn_g/read*
T0*
_output_shapes
: 
w
layer2/conv1/mul_1Mullayer2/conv1/kernel/readlayer2/conv1/mul*
T0*'
_output_shapes
:? 
?
layer2/conv1/Conv2DConv2Dlayer2/Relulayer2/conv1/mul_1*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? 
?
layer2/conv1/BiasAddBiasAddlayer2/conv1/Conv2Dlayer2/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 


layer2/addAddlayer2/conv1/BiasAdd
layer1/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
"layer3/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*$
_class
loc:@layer3/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer3/conv0/wn_g
VariableV2*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer3/conv0/wn_g*
shape:?
?
layer3/conv0/wn_g/AssignAssignlayer3/conv0/wn_g"layer3/conv0/wn_g/Initializer/ones*
T0*$
_class
loc:@layer3/conv0/wn_g*
_output_shapes	
:?
?
layer3/conv0/wn_g/readIdentitylayer3/conv0/wn_g*
T0*$
_class
loc:@layer3/conv0/wn_g*
_output_shapes	
:?
?
4layer3/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *&
_class
loc:@layer3/conv0/kernel*
dtype0*
_output_shapes
:
?
2layer3/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer3/conv0/kernel*
dtype0*
_output_shapes
: 
?
2layer3/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer3/conv0/kernel*
dtype0*
_output_shapes
: 
?
<layer3/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer3/conv0/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@layer3/conv0/kernel*
dtype0*'
_output_shapes
: ?
?
2layer3/conv0/kernel/Initializer/random_uniform/subSub2layer3/conv0/kernel/Initializer/random_uniform/max2layer3/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer3/conv0/kernel*
_output_shapes
: 
?
2layer3/conv0/kernel/Initializer/random_uniform/mulMul<layer3/conv0/kernel/Initializer/random_uniform/RandomUniform2layer3/conv0/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer3/conv0/kernel*'
_output_shapes
: ?
?
.layer3/conv0/kernel/Initializer/random_uniformAdd2layer3/conv0/kernel/Initializer/random_uniform/mul2layer3/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer3/conv0/kernel*'
_output_shapes
: ?
?
layer3/conv0/kernel
VariableV2*&
_class
loc:@layer3/conv0/kernel*
shape: ?*
dtype0*'
_output_shapes
: ?
?
layer3/conv0/kernel/AssignAssignlayer3/conv0/kernel.layer3/conv0/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer3/conv0/kernel*'
_output_shapes
: ?
?
layer3/conv0/kernel/readIdentitylayer3/conv0/kernel*
T0*&
_class
loc:@layer3/conv0/kernel*'
_output_shapes
: ?
?
#layer3/conv0/bias/Initializer/zerosConst*
valueB?*    *$
_class
loc:@layer3/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer3/conv0/bias
VariableV2*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer3/conv0/bias*
shape:?
?
layer3/conv0/bias/AssignAssignlayer3/conv0/bias#layer3/conv0/bias/Initializer/zeros*
T0*$
_class
loc:@layer3/conv0/bias*
_output_shapes	
:?
?
layer3/conv0/bias/readIdentitylayer3/conv0/bias*
T0*$
_class
loc:@layer3/conv0/bias*
_output_shapes	
:?
k
layer3/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer3/conv0/SquareSquarelayer3/conv0/kernel/read*
T0*'
_output_shapes
: ?
w
"layer3/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
v
layer3/conv0/SumSumlayer3/conv0/Square"layer3/conv0/Sum/reduction_indices*
_output_shapes	
:?*
T0
S
layer3/conv0/RsqrtRsqrtlayer3/conv0/Sum*
T0*
_output_shapes	
:?
i
layer3/conv0/mulMullayer3/conv0/Rsqrtlayer3/conv0/wn_g/read*
T0*
_output_shapes	
:?
w
layer3/conv0/mul_1Mullayer3/conv0/kernel/readlayer3/conv0/mul*'
_output_shapes
: ?*
T0
?
layer3/conv0/Conv2DConv2D
layer2/addlayer3/conv0/mul_1*
strides
*
paddingSAME*B
_output_shapes0
.:,????????????????????????????*
T0
?
layer3/conv0/BiasAddBiasAddlayer3/conv0/Conv2Dlayer3/conv0/bias/read*B
_output_shapes0
.:,????????????????????????????*
T0
v
layer3/ReluRelulayer3/conv0/BiasAdd*B
_output_shapes0
.:,????????????????????????????*
T0
?
"layer3/conv1/wn_g/Initializer/onesConst*
dtype0*
_output_shapes
: *
valueB *  ??*$
_class
loc:@layer3/conv1/wn_g

layer3/conv1/wn_g
VariableV2*$
_class
loc:@layer3/conv1/wn_g*
shape: *
dtype0*
_output_shapes
: 
?
layer3/conv1/wn_g/AssignAssignlayer3/conv1/wn_g"layer3/conv1/wn_g/Initializer/ones*
T0*$
_class
loc:@layer3/conv1/wn_g*
_output_shapes
: 
?
layer3/conv1/wn_g/readIdentitylayer3/conv1/wn_g*
T0*$
_class
loc:@layer3/conv1/wn_g*
_output_shapes
: 
?
4layer3/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *&
_class
loc:@layer3/conv1/kernel*
dtype0*
_output_shapes
:
?
2layer3/conv1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?2??*&
_class
loc:@layer3/conv1/kernel
?
2layer3/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer3/conv1/kernel*
dtype0*
_output_shapes
: 
?
<layer3/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer3/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*&
_class
loc:@layer3/conv1/kernel
?
2layer3/conv1/kernel/Initializer/random_uniform/subSub2layer3/conv1/kernel/Initializer/random_uniform/max2layer3/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer3/conv1/kernel*
_output_shapes
: 
?
2layer3/conv1/kernel/Initializer/random_uniform/mulMul<layer3/conv1/kernel/Initializer/random_uniform/RandomUniform2layer3/conv1/kernel/Initializer/random_uniform/sub*'
_output_shapes
:? *
T0*&
_class
loc:@layer3/conv1/kernel
?
.layer3/conv1/kernel/Initializer/random_uniformAdd2layer3/conv1/kernel/Initializer/random_uniform/mul2layer3/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer3/conv1/kernel*'
_output_shapes
:? 
?
layer3/conv1/kernel
VariableV2*
dtype0*'
_output_shapes
:? *&
_class
loc:@layer3/conv1/kernel*
shape:? 
?
layer3/conv1/kernel/AssignAssignlayer3/conv1/kernel.layer3/conv1/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer3/conv1/kernel*'
_output_shapes
:? 
?
layer3/conv1/kernel/readIdentitylayer3/conv1/kernel*
T0*&
_class
loc:@layer3/conv1/kernel*'
_output_shapes
:? 
?
#layer3/conv1/bias/Initializer/zerosConst*
valueB *    *$
_class
loc:@layer3/conv1/bias*
dtype0*
_output_shapes
: 

layer3/conv1/bias
VariableV2*$
_class
loc:@layer3/conv1/bias*
shape: *
dtype0*
_output_shapes
: 
?
layer3/conv1/bias/AssignAssignlayer3/conv1/bias#layer3/conv1/bias/Initializer/zeros*
T0*$
_class
loc:@layer3/conv1/bias*
_output_shapes
: 
?
layer3/conv1/bias/readIdentitylayer3/conv1/bias*
T0*$
_class
loc:@layer3/conv1/bias*
_output_shapes
: 
k
layer3/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
i
layer3/conv1/SquareSquarelayer3/conv1/kernel/read*
T0*'
_output_shapes
:? 
w
"layer3/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
u
layer3/conv1/SumSumlayer3/conv1/Square"layer3/conv1/Sum/reduction_indices*
_output_shapes
: *
T0
R
layer3/conv1/RsqrtRsqrtlayer3/conv1/Sum*
_output_shapes
: *
T0
h
layer3/conv1/mulMullayer3/conv1/Rsqrtlayer3/conv1/wn_g/read*
_output_shapes
: *
T0
w
layer3/conv1/mul_1Mullayer3/conv1/kernel/readlayer3/conv1/mul*
T0*'
_output_shapes
:? 
?
layer3/conv1/Conv2DConv2Dlayer3/Relulayer3/conv1/mul_1*A
_output_shapes/
-:+??????????????????????????? *
T0*
strides
*
paddingSAME
?
layer3/conv1/BiasAddBiasAddlayer3/conv1/Conv2Dlayer3/conv1/bias/read*A
_output_shapes/
-:+??????????????????????????? *
T0


layer3/addAddlayer3/conv1/BiasAdd
layer2/add*A
_output_shapes/
-:+??????????????????????????? *
T0
?
"layer4/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*$
_class
loc:@layer4/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer4/conv0/wn_g
VariableV2*$
_class
loc:@layer4/conv0/wn_g*
shape:?*
dtype0*
_output_shapes	
:?
?
layer4/conv0/wn_g/AssignAssignlayer4/conv0/wn_g"layer4/conv0/wn_g/Initializer/ones*
T0*$
_class
loc:@layer4/conv0/wn_g*
_output_shapes	
:?
?
layer4/conv0/wn_g/readIdentitylayer4/conv0/wn_g*
T0*$
_class
loc:@layer4/conv0/wn_g*
_output_shapes	
:?
?
4layer4/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *&
_class
loc:@layer4/conv0/kernel*
dtype0*
_output_shapes
:
?
2layer4/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer4/conv0/kernel*
dtype0*
_output_shapes
: 
?
2layer4/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer4/conv0/kernel*
dtype0*
_output_shapes
: 
?
<layer4/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer4/conv0/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@layer4/conv0/kernel*
dtype0*'
_output_shapes
: ?
?
2layer4/conv0/kernel/Initializer/random_uniform/subSub2layer4/conv0/kernel/Initializer/random_uniform/max2layer4/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer4/conv0/kernel*
_output_shapes
: 
?
2layer4/conv0/kernel/Initializer/random_uniform/mulMul<layer4/conv0/kernel/Initializer/random_uniform/RandomUniform2layer4/conv0/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer4/conv0/kernel*'
_output_shapes
: ?
?
.layer4/conv0/kernel/Initializer/random_uniformAdd2layer4/conv0/kernel/Initializer/random_uniform/mul2layer4/conv0/kernel/Initializer/random_uniform/min*'
_output_shapes
: ?*
T0*&
_class
loc:@layer4/conv0/kernel
?
layer4/conv0/kernel
VariableV2*
dtype0*'
_output_shapes
: ?*&
_class
loc:@layer4/conv0/kernel*
shape: ?
?
layer4/conv0/kernel/AssignAssignlayer4/conv0/kernel.layer4/conv0/kernel/Initializer/random_uniform*'
_output_shapes
: ?*
T0*&
_class
loc:@layer4/conv0/kernel
?
layer4/conv0/kernel/readIdentitylayer4/conv0/kernel*
T0*&
_class
loc:@layer4/conv0/kernel*'
_output_shapes
: ?
?
#layer4/conv0/bias/Initializer/zerosConst*
valueB?*    *$
_class
loc:@layer4/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer4/conv0/bias
VariableV2*$
_class
loc:@layer4/conv0/bias*
shape:?*
dtype0*
_output_shapes	
:?
?
layer4/conv0/bias/AssignAssignlayer4/conv0/bias#layer4/conv0/bias/Initializer/zeros*
T0*$
_class
loc:@layer4/conv0/bias*
_output_shapes	
:?
?
layer4/conv0/bias/readIdentitylayer4/conv0/bias*
T0*$
_class
loc:@layer4/conv0/bias*
_output_shapes	
:?
k
layer4/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer4/conv0/SquareSquarelayer4/conv0/kernel/read*
T0*'
_output_shapes
: ?
w
"layer4/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
v
layer4/conv0/SumSumlayer4/conv0/Square"layer4/conv0/Sum/reduction_indices*
_output_shapes	
:?*
T0
S
layer4/conv0/RsqrtRsqrtlayer4/conv0/Sum*
T0*
_output_shapes	
:?
i
layer4/conv0/mulMullayer4/conv0/Rsqrtlayer4/conv0/wn_g/read*
T0*
_output_shapes	
:?
w
layer4/conv0/mul_1Mullayer4/conv0/kernel/readlayer4/conv0/mul*
T0*'
_output_shapes
: ?
?
layer4/conv0/Conv2DConv2D
layer3/addlayer4/conv0/mul_1*B
_output_shapes0
.:,????????????????????????????*
T0*
strides
*
paddingSAME
?
layer4/conv0/BiasAddBiasAddlayer4/conv0/Conv2Dlayer4/conv0/bias/read*B
_output_shapes0
.:,????????????????????????????*
T0
v
layer4/ReluRelulayer4/conv0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
"layer4/conv1/wn_g/Initializer/onesConst*
valueB *  ??*$
_class
loc:@layer4/conv1/wn_g*
dtype0*
_output_shapes
: 

layer4/conv1/wn_g
VariableV2*
shape: *
dtype0*
_output_shapes
: *$
_class
loc:@layer4/conv1/wn_g
?
layer4/conv1/wn_g/AssignAssignlayer4/conv1/wn_g"layer4/conv1/wn_g/Initializer/ones*
T0*$
_class
loc:@layer4/conv1/wn_g*
_output_shapes
: 
?
layer4/conv1/wn_g/readIdentitylayer4/conv1/wn_g*
T0*$
_class
loc:@layer4/conv1/wn_g*
_output_shapes
: 
?
4layer4/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *&
_class
loc:@layer4/conv1/kernel*
dtype0*
_output_shapes
:
?
2layer4/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer4/conv1/kernel*
dtype0*
_output_shapes
: 
?
2layer4/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer4/conv1/kernel*
dtype0*
_output_shapes
: 
?
<layer4/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer4/conv1/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@layer4/conv1/kernel*
dtype0*'
_output_shapes
:? 
?
2layer4/conv1/kernel/Initializer/random_uniform/subSub2layer4/conv1/kernel/Initializer/random_uniform/max2layer4/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer4/conv1/kernel*
_output_shapes
: 
?
2layer4/conv1/kernel/Initializer/random_uniform/mulMul<layer4/conv1/kernel/Initializer/random_uniform/RandomUniform2layer4/conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer4/conv1/kernel*'
_output_shapes
:? 
?
.layer4/conv1/kernel/Initializer/random_uniformAdd2layer4/conv1/kernel/Initializer/random_uniform/mul2layer4/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer4/conv1/kernel*'
_output_shapes
:? 
?
layer4/conv1/kernel
VariableV2*
shape:? *
dtype0*'
_output_shapes
:? *&
_class
loc:@layer4/conv1/kernel
?
layer4/conv1/kernel/AssignAssignlayer4/conv1/kernel.layer4/conv1/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer4/conv1/kernel*'
_output_shapes
:? 
?
layer4/conv1/kernel/readIdentitylayer4/conv1/kernel*'
_output_shapes
:? *
T0*&
_class
loc:@layer4/conv1/kernel
?
#layer4/conv1/bias/Initializer/zerosConst*
valueB *    *$
_class
loc:@layer4/conv1/bias*
dtype0*
_output_shapes
: 

layer4/conv1/bias
VariableV2*
dtype0*
_output_shapes
: *$
_class
loc:@layer4/conv1/bias*
shape: 
?
layer4/conv1/bias/AssignAssignlayer4/conv1/bias#layer4/conv1/bias/Initializer/zeros*
T0*$
_class
loc:@layer4/conv1/bias*
_output_shapes
: 
?
layer4/conv1/bias/readIdentitylayer4/conv1/bias*
T0*$
_class
loc:@layer4/conv1/bias*
_output_shapes
: 
k
layer4/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer4/conv1/SquareSquarelayer4/conv1/kernel/read*
T0*'
_output_shapes
:? 
w
"layer4/conv1/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
u
layer4/conv1/SumSumlayer4/conv1/Square"layer4/conv1/Sum/reduction_indices*
_output_shapes
: *
T0
R
layer4/conv1/RsqrtRsqrtlayer4/conv1/Sum*
_output_shapes
: *
T0
h
layer4/conv1/mulMullayer4/conv1/Rsqrtlayer4/conv1/wn_g/read*
T0*
_output_shapes
: 
w
layer4/conv1/mul_1Mullayer4/conv1/kernel/readlayer4/conv1/mul*'
_output_shapes
:? *
T0
?
layer4/conv1/Conv2DConv2Dlayer4/Relulayer4/conv1/mul_1*A
_output_shapes/
-:+??????????????????????????? *
T0*
strides
*
paddingSAME
?
layer4/conv1/BiasAddBiasAddlayer4/conv1/Conv2Dlayer4/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 


layer4/addAddlayer4/conv1/BiasAdd
layer3/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
"layer5/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*$
_class
loc:@layer5/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer5/conv0/wn_g
VariableV2*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer5/conv0/wn_g*
shape:?
?
layer5/conv0/wn_g/AssignAssignlayer5/conv0/wn_g"layer5/conv0/wn_g/Initializer/ones*
T0*$
_class
loc:@layer5/conv0/wn_g*
_output_shapes	
:?
?
layer5/conv0/wn_g/readIdentitylayer5/conv0/wn_g*
T0*$
_class
loc:@layer5/conv0/wn_g*
_output_shapes	
:?
?
4layer5/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *&
_class
loc:@layer5/conv0/kernel*
dtype0*
_output_shapes
:
?
2layer5/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer5/conv0/kernel*
dtype0*
_output_shapes
: 
?
2layer5/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer5/conv0/kernel*
dtype0*
_output_shapes
: 
?
<layer5/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer5/conv0/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
: ?*
T0*&
_class
loc:@layer5/conv0/kernel
?
2layer5/conv0/kernel/Initializer/random_uniform/subSub2layer5/conv0/kernel/Initializer/random_uniform/max2layer5/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer5/conv0/kernel*
_output_shapes
: 
?
2layer5/conv0/kernel/Initializer/random_uniform/mulMul<layer5/conv0/kernel/Initializer/random_uniform/RandomUniform2layer5/conv0/kernel/Initializer/random_uniform/sub*'
_output_shapes
: ?*
T0*&
_class
loc:@layer5/conv0/kernel
?
.layer5/conv0/kernel/Initializer/random_uniformAdd2layer5/conv0/kernel/Initializer/random_uniform/mul2layer5/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer5/conv0/kernel*'
_output_shapes
: ?
?
layer5/conv0/kernel
VariableV2*&
_class
loc:@layer5/conv0/kernel*
shape: ?*
dtype0*'
_output_shapes
: ?
?
layer5/conv0/kernel/AssignAssignlayer5/conv0/kernel.layer5/conv0/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer5/conv0/kernel*'
_output_shapes
: ?
?
layer5/conv0/kernel/readIdentitylayer5/conv0/kernel*
T0*&
_class
loc:@layer5/conv0/kernel*'
_output_shapes
: ?
?
#layer5/conv0/bias/Initializer/zerosConst*
valueB?*    *$
_class
loc:@layer5/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer5/conv0/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer5/conv0/bias
?
layer5/conv0/bias/AssignAssignlayer5/conv0/bias#layer5/conv0/bias/Initializer/zeros*
T0*$
_class
loc:@layer5/conv0/bias*
_output_shapes	
:?
?
layer5/conv0/bias/readIdentitylayer5/conv0/bias*
T0*$
_class
loc:@layer5/conv0/bias*
_output_shapes	
:?
k
layer5/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer5/conv0/SquareSquarelayer5/conv0/kernel/read*
T0*'
_output_shapes
: ?
w
"layer5/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
v
layer5/conv0/SumSumlayer5/conv0/Square"layer5/conv0/Sum/reduction_indices*
T0*
_output_shapes	
:?
S
layer5/conv0/RsqrtRsqrtlayer5/conv0/Sum*
T0*
_output_shapes	
:?
i
layer5/conv0/mulMullayer5/conv0/Rsqrtlayer5/conv0/wn_g/read*
T0*
_output_shapes	
:?
w
layer5/conv0/mul_1Mullayer5/conv0/kernel/readlayer5/conv0/mul*
T0*'
_output_shapes
: ?
?
layer5/conv0/Conv2DConv2D
layer4/addlayer5/conv0/mul_1*B
_output_shapes0
.:,????????????????????????????*
T0*
strides
*
paddingSAME
?
layer5/conv0/BiasAddBiasAddlayer5/conv0/Conv2Dlayer5/conv0/bias/read*
T0*B
_output_shapes0
.:,????????????????????????????
v
layer5/ReluRelulayer5/conv0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
"layer5/conv1/wn_g/Initializer/onesConst*
valueB *  ??*$
_class
loc:@layer5/conv1/wn_g*
dtype0*
_output_shapes
: 

layer5/conv1/wn_g
VariableV2*$
_class
loc:@layer5/conv1/wn_g*
shape: *
dtype0*
_output_shapes
: 
?
layer5/conv1/wn_g/AssignAssignlayer5/conv1/wn_g"layer5/conv1/wn_g/Initializer/ones*
T0*$
_class
loc:@layer5/conv1/wn_g*
_output_shapes
: 
?
layer5/conv1/wn_g/readIdentitylayer5/conv1/wn_g*
T0*$
_class
loc:@layer5/conv1/wn_g*
_output_shapes
: 
?
4layer5/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *&
_class
loc:@layer5/conv1/kernel*
dtype0*
_output_shapes
:
?
2layer5/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer5/conv1/kernel*
dtype0*
_output_shapes
: 
?
2layer5/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer5/conv1/kernel*
dtype0*
_output_shapes
: 
?
<layer5/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer5/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*&
_class
loc:@layer5/conv1/kernel
?
2layer5/conv1/kernel/Initializer/random_uniform/subSub2layer5/conv1/kernel/Initializer/random_uniform/max2layer5/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer5/conv1/kernel*
_output_shapes
: 
?
2layer5/conv1/kernel/Initializer/random_uniform/mulMul<layer5/conv1/kernel/Initializer/random_uniform/RandomUniform2layer5/conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer5/conv1/kernel*'
_output_shapes
:? 
?
.layer5/conv1/kernel/Initializer/random_uniformAdd2layer5/conv1/kernel/Initializer/random_uniform/mul2layer5/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer5/conv1/kernel*'
_output_shapes
:? 
?
layer5/conv1/kernel
VariableV2*&
_class
loc:@layer5/conv1/kernel*
shape:? *
dtype0*'
_output_shapes
:? 
?
layer5/conv1/kernel/AssignAssignlayer5/conv1/kernel.layer5/conv1/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer5/conv1/kernel*'
_output_shapes
:? 
?
layer5/conv1/kernel/readIdentitylayer5/conv1/kernel*'
_output_shapes
:? *
T0*&
_class
loc:@layer5/conv1/kernel
?
#layer5/conv1/bias/Initializer/zerosConst*
valueB *    *$
_class
loc:@layer5/conv1/bias*
dtype0*
_output_shapes
: 

layer5/conv1/bias
VariableV2*
shape: *
dtype0*
_output_shapes
: *$
_class
loc:@layer5/conv1/bias
?
layer5/conv1/bias/AssignAssignlayer5/conv1/bias#layer5/conv1/bias/Initializer/zeros*
T0*$
_class
loc:@layer5/conv1/bias*
_output_shapes
: 
?
layer5/conv1/bias/readIdentitylayer5/conv1/bias*
T0*$
_class
loc:@layer5/conv1/bias*
_output_shapes
: 
k
layer5/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer5/conv1/SquareSquarelayer5/conv1/kernel/read*'
_output_shapes
:? *
T0
w
"layer5/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
u
layer5/conv1/SumSumlayer5/conv1/Square"layer5/conv1/Sum/reduction_indices*
_output_shapes
: *
T0
R
layer5/conv1/RsqrtRsqrtlayer5/conv1/Sum*
T0*
_output_shapes
: 
h
layer5/conv1/mulMullayer5/conv1/Rsqrtlayer5/conv1/wn_g/read*
T0*
_output_shapes
: 
w
layer5/conv1/mul_1Mullayer5/conv1/kernel/readlayer5/conv1/mul*
T0*'
_output_shapes
:? 
?
layer5/conv1/Conv2DConv2Dlayer5/Relulayer5/conv1/mul_1*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? 
?
layer5/conv1/BiasAddBiasAddlayer5/conv1/Conv2Dlayer5/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 


layer5/addAddlayer5/conv1/BiasAdd
layer4/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
"layer6/conv0/wn_g/Initializer/onesConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??*$
_class
loc:@layer6/conv0/wn_g
?
layer6/conv0/wn_g
VariableV2*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer6/conv0/wn_g*
shape:?
?
layer6/conv0/wn_g/AssignAssignlayer6/conv0/wn_g"layer6/conv0/wn_g/Initializer/ones*
_output_shapes	
:?*
T0*$
_class
loc:@layer6/conv0/wn_g
?
layer6/conv0/wn_g/readIdentitylayer6/conv0/wn_g*
T0*$
_class
loc:@layer6/conv0/wn_g*
_output_shapes	
:?
?
4layer6/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *&
_class
loc:@layer6/conv0/kernel*
dtype0*
_output_shapes
:
?
2layer6/conv0/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?2??*&
_class
loc:@layer6/conv0/kernel
?
2layer6/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer6/conv0/kernel*
dtype0*
_output_shapes
: 
?
<layer6/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer6/conv0/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
: ?*
T0*&
_class
loc:@layer6/conv0/kernel
?
2layer6/conv0/kernel/Initializer/random_uniform/subSub2layer6/conv0/kernel/Initializer/random_uniform/max2layer6/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer6/conv0/kernel*
_output_shapes
: 
?
2layer6/conv0/kernel/Initializer/random_uniform/mulMul<layer6/conv0/kernel/Initializer/random_uniform/RandomUniform2layer6/conv0/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer6/conv0/kernel*'
_output_shapes
: ?
?
.layer6/conv0/kernel/Initializer/random_uniformAdd2layer6/conv0/kernel/Initializer/random_uniform/mul2layer6/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer6/conv0/kernel*'
_output_shapes
: ?
?
layer6/conv0/kernel
VariableV2*&
_class
loc:@layer6/conv0/kernel*
shape: ?*
dtype0*'
_output_shapes
: ?
?
layer6/conv0/kernel/AssignAssignlayer6/conv0/kernel.layer6/conv0/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer6/conv0/kernel*'
_output_shapes
: ?
?
layer6/conv0/kernel/readIdentitylayer6/conv0/kernel*
T0*&
_class
loc:@layer6/conv0/kernel*'
_output_shapes
: ?
?
#layer6/conv0/bias/Initializer/zerosConst*
valueB?*    *$
_class
loc:@layer6/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer6/conv0/bias
VariableV2*$
_class
loc:@layer6/conv0/bias*
shape:?*
dtype0*
_output_shapes	
:?
?
layer6/conv0/bias/AssignAssignlayer6/conv0/bias#layer6/conv0/bias/Initializer/zeros*
T0*$
_class
loc:@layer6/conv0/bias*
_output_shapes	
:?
?
layer6/conv0/bias/readIdentitylayer6/conv0/bias*
T0*$
_class
loc:@layer6/conv0/bias*
_output_shapes	
:?
k
layer6/conv0/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
i
layer6/conv0/SquareSquarelayer6/conv0/kernel/read*
T0*'
_output_shapes
: ?
w
"layer6/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
v
layer6/conv0/SumSumlayer6/conv0/Square"layer6/conv0/Sum/reduction_indices*
_output_shapes	
:?*
T0
S
layer6/conv0/RsqrtRsqrtlayer6/conv0/Sum*
T0*
_output_shapes	
:?
i
layer6/conv0/mulMullayer6/conv0/Rsqrtlayer6/conv0/wn_g/read*
T0*
_output_shapes	
:?
w
layer6/conv0/mul_1Mullayer6/conv0/kernel/readlayer6/conv0/mul*
T0*'
_output_shapes
: ?
?
layer6/conv0/Conv2DConv2D
layer5/addlayer6/conv0/mul_1*B
_output_shapes0
.:,????????????????????????????*
T0*
strides
*
paddingSAME
?
layer6/conv0/BiasAddBiasAddlayer6/conv0/Conv2Dlayer6/conv0/bias/read*
T0*B
_output_shapes0
.:,????????????????????????????
v
layer6/ReluRelulayer6/conv0/BiasAdd*B
_output_shapes0
.:,????????????????????????????*
T0
?
"layer6/conv1/wn_g/Initializer/onesConst*
valueB *  ??*$
_class
loc:@layer6/conv1/wn_g*
dtype0*
_output_shapes
: 

layer6/conv1/wn_g
VariableV2*$
_class
loc:@layer6/conv1/wn_g*
shape: *
dtype0*
_output_shapes
: 
?
layer6/conv1/wn_g/AssignAssignlayer6/conv1/wn_g"layer6/conv1/wn_g/Initializer/ones*
_output_shapes
: *
T0*$
_class
loc:@layer6/conv1/wn_g
?
layer6/conv1/wn_g/readIdentitylayer6/conv1/wn_g*
T0*$
_class
loc:@layer6/conv1/wn_g*
_output_shapes
: 
?
4layer6/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *&
_class
loc:@layer6/conv1/kernel*
dtype0*
_output_shapes
:
?
2layer6/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer6/conv1/kernel*
dtype0*
_output_shapes
: 
?
2layer6/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer6/conv1/kernel*
dtype0*
_output_shapes
: 
?
<layer6/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer6/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*&
_class
loc:@layer6/conv1/kernel
?
2layer6/conv1/kernel/Initializer/random_uniform/subSub2layer6/conv1/kernel/Initializer/random_uniform/max2layer6/conv1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*&
_class
loc:@layer6/conv1/kernel
?
2layer6/conv1/kernel/Initializer/random_uniform/mulMul<layer6/conv1/kernel/Initializer/random_uniform/RandomUniform2layer6/conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer6/conv1/kernel*'
_output_shapes
:? 
?
.layer6/conv1/kernel/Initializer/random_uniformAdd2layer6/conv1/kernel/Initializer/random_uniform/mul2layer6/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer6/conv1/kernel*'
_output_shapes
:? 
?
layer6/conv1/kernel
VariableV2*&
_class
loc:@layer6/conv1/kernel*
shape:? *
dtype0*'
_output_shapes
:? 
?
layer6/conv1/kernel/AssignAssignlayer6/conv1/kernel.layer6/conv1/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer6/conv1/kernel*'
_output_shapes
:? 
?
layer6/conv1/kernel/readIdentitylayer6/conv1/kernel*
T0*&
_class
loc:@layer6/conv1/kernel*'
_output_shapes
:? 
?
#layer6/conv1/bias/Initializer/zerosConst*
valueB *    *$
_class
loc:@layer6/conv1/bias*
dtype0*
_output_shapes
: 

layer6/conv1/bias
VariableV2*
dtype0*
_output_shapes
: *$
_class
loc:@layer6/conv1/bias*
shape: 
?
layer6/conv1/bias/AssignAssignlayer6/conv1/bias#layer6/conv1/bias/Initializer/zeros*
_output_shapes
: *
T0*$
_class
loc:@layer6/conv1/bias
?
layer6/conv1/bias/readIdentitylayer6/conv1/bias*
_output_shapes
: *
T0*$
_class
loc:@layer6/conv1/bias
k
layer6/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer6/conv1/SquareSquarelayer6/conv1/kernel/read*'
_output_shapes
:? *
T0
w
"layer6/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
u
layer6/conv1/SumSumlayer6/conv1/Square"layer6/conv1/Sum/reduction_indices*
T0*
_output_shapes
: 
R
layer6/conv1/RsqrtRsqrtlayer6/conv1/Sum*
T0*
_output_shapes
: 
h
layer6/conv1/mulMullayer6/conv1/Rsqrtlayer6/conv1/wn_g/read*
T0*
_output_shapes
: 
w
layer6/conv1/mul_1Mullayer6/conv1/kernel/readlayer6/conv1/mul*
T0*'
_output_shapes
:? 
?
layer6/conv1/Conv2DConv2Dlayer6/Relulayer6/conv1/mul_1*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? 
?
layer6/conv1/BiasAddBiasAddlayer6/conv1/Conv2Dlayer6/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 


layer6/addAddlayer6/conv1/BiasAdd
layer5/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
"layer7/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*$
_class
loc:@layer7/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer7/conv0/wn_g
VariableV2*$
_class
loc:@layer7/conv0/wn_g*
shape:?*
dtype0*
_output_shapes	
:?
?
layer7/conv0/wn_g/AssignAssignlayer7/conv0/wn_g"layer7/conv0/wn_g/Initializer/ones*
T0*$
_class
loc:@layer7/conv0/wn_g*
_output_shapes	
:?
?
layer7/conv0/wn_g/readIdentitylayer7/conv0/wn_g*
T0*$
_class
loc:@layer7/conv0/wn_g*
_output_shapes	
:?
?
4layer7/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *&
_class
loc:@layer7/conv0/kernel*
dtype0*
_output_shapes
:
?
2layer7/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer7/conv0/kernel*
dtype0*
_output_shapes
: 
?
2layer7/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer7/conv0/kernel*
dtype0*
_output_shapes
: 
?
<layer7/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer7/conv0/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
: ?*
T0*&
_class
loc:@layer7/conv0/kernel
?
2layer7/conv0/kernel/Initializer/random_uniform/subSub2layer7/conv0/kernel/Initializer/random_uniform/max2layer7/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer7/conv0/kernel*
_output_shapes
: 
?
2layer7/conv0/kernel/Initializer/random_uniform/mulMul<layer7/conv0/kernel/Initializer/random_uniform/RandomUniform2layer7/conv0/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer7/conv0/kernel*'
_output_shapes
: ?
?
.layer7/conv0/kernel/Initializer/random_uniformAdd2layer7/conv0/kernel/Initializer/random_uniform/mul2layer7/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer7/conv0/kernel*'
_output_shapes
: ?
?
layer7/conv0/kernel
VariableV2*
dtype0*'
_output_shapes
: ?*&
_class
loc:@layer7/conv0/kernel*
shape: ?
?
layer7/conv0/kernel/AssignAssignlayer7/conv0/kernel.layer7/conv0/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer7/conv0/kernel*'
_output_shapes
: ?
?
layer7/conv0/kernel/readIdentitylayer7/conv0/kernel*
T0*&
_class
loc:@layer7/conv0/kernel*'
_output_shapes
: ?
?
#layer7/conv0/bias/Initializer/zerosConst*
valueB?*    *$
_class
loc:@layer7/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer7/conv0/bias
VariableV2*$
_class
loc:@layer7/conv0/bias*
shape:?*
dtype0*
_output_shapes	
:?
?
layer7/conv0/bias/AssignAssignlayer7/conv0/bias#layer7/conv0/bias/Initializer/zeros*
_output_shapes	
:?*
T0*$
_class
loc:@layer7/conv0/bias
?
layer7/conv0/bias/readIdentitylayer7/conv0/bias*
T0*$
_class
loc:@layer7/conv0/bias*
_output_shapes	
:?
k
layer7/conv0/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
i
layer7/conv0/SquareSquarelayer7/conv0/kernel/read*
T0*'
_output_shapes
: ?
w
"layer7/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
v
layer7/conv0/SumSumlayer7/conv0/Square"layer7/conv0/Sum/reduction_indices*
T0*
_output_shapes	
:?
S
layer7/conv0/RsqrtRsqrtlayer7/conv0/Sum*
T0*
_output_shapes	
:?
i
layer7/conv0/mulMullayer7/conv0/Rsqrtlayer7/conv0/wn_g/read*
_output_shapes	
:?*
T0
w
layer7/conv0/mul_1Mullayer7/conv0/kernel/readlayer7/conv0/mul*
T0*'
_output_shapes
: ?
?
layer7/conv0/Conv2DConv2D
layer6/addlayer7/conv0/mul_1*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,????????????????????????????
?
layer7/conv0/BiasAddBiasAddlayer7/conv0/Conv2Dlayer7/conv0/bias/read*
T0*B
_output_shapes0
.:,????????????????????????????
v
layer7/ReluRelulayer7/conv0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
"layer7/conv1/wn_g/Initializer/onesConst*
dtype0*
_output_shapes
: *
valueB *  ??*$
_class
loc:@layer7/conv1/wn_g

layer7/conv1/wn_g
VariableV2*
dtype0*
_output_shapes
: *$
_class
loc:@layer7/conv1/wn_g*
shape: 
?
layer7/conv1/wn_g/AssignAssignlayer7/conv1/wn_g"layer7/conv1/wn_g/Initializer/ones*
T0*$
_class
loc:@layer7/conv1/wn_g*
_output_shapes
: 
?
layer7/conv1/wn_g/readIdentitylayer7/conv1/wn_g*
T0*$
_class
loc:@layer7/conv1/wn_g*
_output_shapes
: 
?
4layer7/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *&
_class
loc:@layer7/conv1/kernel*
dtype0*
_output_shapes
:
?
2layer7/conv1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?2??*&
_class
loc:@layer7/conv1/kernel
?
2layer7/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer7/conv1/kernel*
dtype0*
_output_shapes
: 
?
<layer7/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer7/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*&
_class
loc:@layer7/conv1/kernel
?
2layer7/conv1/kernel/Initializer/random_uniform/subSub2layer7/conv1/kernel/Initializer/random_uniform/max2layer7/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer7/conv1/kernel*
_output_shapes
: 
?
2layer7/conv1/kernel/Initializer/random_uniform/mulMul<layer7/conv1/kernel/Initializer/random_uniform/RandomUniform2layer7/conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer7/conv1/kernel*'
_output_shapes
:? 
?
.layer7/conv1/kernel/Initializer/random_uniformAdd2layer7/conv1/kernel/Initializer/random_uniform/mul2layer7/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer7/conv1/kernel*'
_output_shapes
:? 
?
layer7/conv1/kernel
VariableV2*
shape:? *
dtype0*'
_output_shapes
:? *&
_class
loc:@layer7/conv1/kernel
?
layer7/conv1/kernel/AssignAssignlayer7/conv1/kernel.layer7/conv1/kernel/Initializer/random_uniform*'
_output_shapes
:? *
T0*&
_class
loc:@layer7/conv1/kernel
?
layer7/conv1/kernel/readIdentitylayer7/conv1/kernel*'
_output_shapes
:? *
T0*&
_class
loc:@layer7/conv1/kernel
?
#layer7/conv1/bias/Initializer/zerosConst*
valueB *    *$
_class
loc:@layer7/conv1/bias*
dtype0*
_output_shapes
: 

layer7/conv1/bias
VariableV2*$
_class
loc:@layer7/conv1/bias*
shape: *
dtype0*
_output_shapes
: 
?
layer7/conv1/bias/AssignAssignlayer7/conv1/bias#layer7/conv1/bias/Initializer/zeros*
T0*$
_class
loc:@layer7/conv1/bias*
_output_shapes
: 
?
layer7/conv1/bias/readIdentitylayer7/conv1/bias*
T0*$
_class
loc:@layer7/conv1/bias*
_output_shapes
: 
k
layer7/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer7/conv1/SquareSquarelayer7/conv1/kernel/read*
T0*'
_output_shapes
:? 
w
"layer7/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
u
layer7/conv1/SumSumlayer7/conv1/Square"layer7/conv1/Sum/reduction_indices*
_output_shapes
: *
T0
R
layer7/conv1/RsqrtRsqrtlayer7/conv1/Sum*
T0*
_output_shapes
: 
h
layer7/conv1/mulMullayer7/conv1/Rsqrtlayer7/conv1/wn_g/read*
T0*
_output_shapes
: 
w
layer7/conv1/mul_1Mullayer7/conv1/kernel/readlayer7/conv1/mul*'
_output_shapes
:? *
T0
?
layer7/conv1/Conv2DConv2Dlayer7/Relulayer7/conv1/mul_1*A
_output_shapes/
-:+??????????????????????????? *
T0*
strides
*
paddingSAME
?
layer7/conv1/BiasAddBiasAddlayer7/conv1/Conv2Dlayer7/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 


layer7/addAddlayer7/conv1/BiasAdd
layer6/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
"layer8/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*$
_class
loc:@layer8/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer8/conv0/wn_g
VariableV2*$
_class
loc:@layer8/conv0/wn_g*
shape:?*
dtype0*
_output_shapes	
:?
?
layer8/conv0/wn_g/AssignAssignlayer8/conv0/wn_g"layer8/conv0/wn_g/Initializer/ones*
T0*$
_class
loc:@layer8/conv0/wn_g*
_output_shapes	
:?
?
layer8/conv0/wn_g/readIdentitylayer8/conv0/wn_g*
_output_shapes	
:?*
T0*$
_class
loc:@layer8/conv0/wn_g
?
4layer8/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *&
_class
loc:@layer8/conv0/kernel*
dtype0*
_output_shapes
:
?
2layer8/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer8/conv0/kernel*
dtype0*
_output_shapes
: 
?
2layer8/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer8/conv0/kernel*
dtype0*
_output_shapes
: 
?
<layer8/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer8/conv0/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@layer8/conv0/kernel*
dtype0*'
_output_shapes
: ?
?
2layer8/conv0/kernel/Initializer/random_uniform/subSub2layer8/conv0/kernel/Initializer/random_uniform/max2layer8/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer8/conv0/kernel*
_output_shapes
: 
?
2layer8/conv0/kernel/Initializer/random_uniform/mulMul<layer8/conv0/kernel/Initializer/random_uniform/RandomUniform2layer8/conv0/kernel/Initializer/random_uniform/sub*'
_output_shapes
: ?*
T0*&
_class
loc:@layer8/conv0/kernel
?
.layer8/conv0/kernel/Initializer/random_uniformAdd2layer8/conv0/kernel/Initializer/random_uniform/mul2layer8/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer8/conv0/kernel*'
_output_shapes
: ?
?
layer8/conv0/kernel
VariableV2*
shape: ?*
dtype0*'
_output_shapes
: ?*&
_class
loc:@layer8/conv0/kernel
?
layer8/conv0/kernel/AssignAssignlayer8/conv0/kernel.layer8/conv0/kernel/Initializer/random_uniform*'
_output_shapes
: ?*
T0*&
_class
loc:@layer8/conv0/kernel
?
layer8/conv0/kernel/readIdentitylayer8/conv0/kernel*
T0*&
_class
loc:@layer8/conv0/kernel*'
_output_shapes
: ?
?
#layer8/conv0/bias/Initializer/zerosConst*
valueB?*    *$
_class
loc:@layer8/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer8/conv0/bias
VariableV2*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer8/conv0/bias*
shape:?
?
layer8/conv0/bias/AssignAssignlayer8/conv0/bias#layer8/conv0/bias/Initializer/zeros*
T0*$
_class
loc:@layer8/conv0/bias*
_output_shapes	
:?
?
layer8/conv0/bias/readIdentitylayer8/conv0/bias*
T0*$
_class
loc:@layer8/conv0/bias*
_output_shapes	
:?
k
layer8/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer8/conv0/SquareSquarelayer8/conv0/kernel/read*
T0*'
_output_shapes
: ?
w
"layer8/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
v
layer8/conv0/SumSumlayer8/conv0/Square"layer8/conv0/Sum/reduction_indices*
_output_shapes	
:?*
T0
S
layer8/conv0/RsqrtRsqrtlayer8/conv0/Sum*
T0*
_output_shapes	
:?
i
layer8/conv0/mulMullayer8/conv0/Rsqrtlayer8/conv0/wn_g/read*
T0*
_output_shapes	
:?
w
layer8/conv0/mul_1Mullayer8/conv0/kernel/readlayer8/conv0/mul*
T0*'
_output_shapes
: ?
?
layer8/conv0/Conv2DConv2D
layer7/addlayer8/conv0/mul_1*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,????????????????????????????
?
layer8/conv0/BiasAddBiasAddlayer8/conv0/Conv2Dlayer8/conv0/bias/read*
T0*B
_output_shapes0
.:,????????????????????????????
v
layer8/ReluRelulayer8/conv0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
"layer8/conv1/wn_g/Initializer/onesConst*
dtype0*
_output_shapes
: *
valueB *  ??*$
_class
loc:@layer8/conv1/wn_g

layer8/conv1/wn_g
VariableV2*$
_class
loc:@layer8/conv1/wn_g*
shape: *
dtype0*
_output_shapes
: 
?
layer8/conv1/wn_g/AssignAssignlayer8/conv1/wn_g"layer8/conv1/wn_g/Initializer/ones*
T0*$
_class
loc:@layer8/conv1/wn_g*
_output_shapes
: 
?
layer8/conv1/wn_g/readIdentitylayer8/conv1/wn_g*
_output_shapes
: *
T0*$
_class
loc:@layer8/conv1/wn_g
?
4layer8/conv1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      ?       *&
_class
loc:@layer8/conv1/kernel
?
2layer8/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer8/conv1/kernel*
dtype0*
_output_shapes
: 
?
2layer8/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer8/conv1/kernel*
dtype0*
_output_shapes
: 
?
<layer8/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer8/conv1/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@layer8/conv1/kernel*
dtype0*'
_output_shapes
:? 
?
2layer8/conv1/kernel/Initializer/random_uniform/subSub2layer8/conv1/kernel/Initializer/random_uniform/max2layer8/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer8/conv1/kernel*
_output_shapes
: 
?
2layer8/conv1/kernel/Initializer/random_uniform/mulMul<layer8/conv1/kernel/Initializer/random_uniform/RandomUniform2layer8/conv1/kernel/Initializer/random_uniform/sub*'
_output_shapes
:? *
T0*&
_class
loc:@layer8/conv1/kernel
?
.layer8/conv1/kernel/Initializer/random_uniformAdd2layer8/conv1/kernel/Initializer/random_uniform/mul2layer8/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer8/conv1/kernel*'
_output_shapes
:? 
?
layer8/conv1/kernel
VariableV2*
dtype0*'
_output_shapes
:? *&
_class
loc:@layer8/conv1/kernel*
shape:? 
?
layer8/conv1/kernel/AssignAssignlayer8/conv1/kernel.layer8/conv1/kernel/Initializer/random_uniform*'
_output_shapes
:? *
T0*&
_class
loc:@layer8/conv1/kernel
?
layer8/conv1/kernel/readIdentitylayer8/conv1/kernel*
T0*&
_class
loc:@layer8/conv1/kernel*'
_output_shapes
:? 
?
#layer8/conv1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *$
_class
loc:@layer8/conv1/bias

layer8/conv1/bias
VariableV2*$
_class
loc:@layer8/conv1/bias*
shape: *
dtype0*
_output_shapes
: 
?
layer8/conv1/bias/AssignAssignlayer8/conv1/bias#layer8/conv1/bias/Initializer/zeros*
T0*$
_class
loc:@layer8/conv1/bias*
_output_shapes
: 
?
layer8/conv1/bias/readIdentitylayer8/conv1/bias*
T0*$
_class
loc:@layer8/conv1/bias*
_output_shapes
: 
k
layer8/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer8/conv1/SquareSquarelayer8/conv1/kernel/read*
T0*'
_output_shapes
:? 
w
"layer8/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
u
layer8/conv1/SumSumlayer8/conv1/Square"layer8/conv1/Sum/reduction_indices*
T0*
_output_shapes
: 
R
layer8/conv1/RsqrtRsqrtlayer8/conv1/Sum*
T0*
_output_shapes
: 
h
layer8/conv1/mulMullayer8/conv1/Rsqrtlayer8/conv1/wn_g/read*
T0*
_output_shapes
: 
w
layer8/conv1/mul_1Mullayer8/conv1/kernel/readlayer8/conv1/mul*
T0*'
_output_shapes
:? 
?
layer8/conv1/Conv2DConv2Dlayer8/Relulayer8/conv1/mul_1*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? 
?
layer8/conv1/BiasAddBiasAddlayer8/conv1/Conv2Dlayer8/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 


layer8/addAddlayer8/conv1/BiasAdd
layer7/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
"layer9/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*$
_class
loc:@layer9/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer9/conv0/wn_g
VariableV2*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer9/conv0/wn_g*
shape:?
?
layer9/conv0/wn_g/AssignAssignlayer9/conv0/wn_g"layer9/conv0/wn_g/Initializer/ones*
T0*$
_class
loc:@layer9/conv0/wn_g*
_output_shapes	
:?
?
layer9/conv0/wn_g/readIdentitylayer9/conv0/wn_g*
T0*$
_class
loc:@layer9/conv0/wn_g*
_output_shapes	
:?
?
4layer9/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *&
_class
loc:@layer9/conv0/kernel*
dtype0*
_output_shapes
:
?
2layer9/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer9/conv0/kernel*
dtype0*
_output_shapes
: 
?
2layer9/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer9/conv0/kernel*
dtype0*
_output_shapes
: 
?
<layer9/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer9/conv0/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
: ?*
T0*&
_class
loc:@layer9/conv0/kernel
?
2layer9/conv0/kernel/Initializer/random_uniform/subSub2layer9/conv0/kernel/Initializer/random_uniform/max2layer9/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer9/conv0/kernel*
_output_shapes
: 
?
2layer9/conv0/kernel/Initializer/random_uniform/mulMul<layer9/conv0/kernel/Initializer/random_uniform/RandomUniform2layer9/conv0/kernel/Initializer/random_uniform/sub*'
_output_shapes
: ?*
T0*&
_class
loc:@layer9/conv0/kernel
?
.layer9/conv0/kernel/Initializer/random_uniformAdd2layer9/conv0/kernel/Initializer/random_uniform/mul2layer9/conv0/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer9/conv0/kernel*'
_output_shapes
: ?
?
layer9/conv0/kernel
VariableV2*
dtype0*'
_output_shapes
: ?*&
_class
loc:@layer9/conv0/kernel*
shape: ?
?
layer9/conv0/kernel/AssignAssignlayer9/conv0/kernel.layer9/conv0/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer9/conv0/kernel*'
_output_shapes
: ?
?
layer9/conv0/kernel/readIdentitylayer9/conv0/kernel*
T0*&
_class
loc:@layer9/conv0/kernel*'
_output_shapes
: ?
?
#layer9/conv0/bias/Initializer/zerosConst*
valueB?*    *$
_class
loc:@layer9/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer9/conv0/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*$
_class
loc:@layer9/conv0/bias
?
layer9/conv0/bias/AssignAssignlayer9/conv0/bias#layer9/conv0/bias/Initializer/zeros*
T0*$
_class
loc:@layer9/conv0/bias*
_output_shapes	
:?
?
layer9/conv0/bias/readIdentitylayer9/conv0/bias*
T0*$
_class
loc:@layer9/conv0/bias*
_output_shapes	
:?
k
layer9/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer9/conv0/SquareSquarelayer9/conv0/kernel/read*
T0*'
_output_shapes
: ?
w
"layer9/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
v
layer9/conv0/SumSumlayer9/conv0/Square"layer9/conv0/Sum/reduction_indices*
T0*
_output_shapes	
:?
S
layer9/conv0/RsqrtRsqrtlayer9/conv0/Sum*
T0*
_output_shapes	
:?
i
layer9/conv0/mulMullayer9/conv0/Rsqrtlayer9/conv0/wn_g/read*
T0*
_output_shapes	
:?
w
layer9/conv0/mul_1Mullayer9/conv0/kernel/readlayer9/conv0/mul*'
_output_shapes
: ?*
T0
?
layer9/conv0/Conv2DConv2D
layer8/addlayer9/conv0/mul_1*
paddingSAME*B
_output_shapes0
.:,????????????????????????????*
T0*
strides

?
layer9/conv0/BiasAddBiasAddlayer9/conv0/Conv2Dlayer9/conv0/bias/read*
T0*B
_output_shapes0
.:,????????????????????????????
v
layer9/ReluRelulayer9/conv0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
"layer9/conv1/wn_g/Initializer/onesConst*
valueB *  ??*$
_class
loc:@layer9/conv1/wn_g*
dtype0*
_output_shapes
: 

layer9/conv1/wn_g
VariableV2*
dtype0*
_output_shapes
: *$
_class
loc:@layer9/conv1/wn_g*
shape: 
?
layer9/conv1/wn_g/AssignAssignlayer9/conv1/wn_g"layer9/conv1/wn_g/Initializer/ones*
_output_shapes
: *
T0*$
_class
loc:@layer9/conv1/wn_g
?
layer9/conv1/wn_g/readIdentitylayer9/conv1/wn_g*
T0*$
_class
loc:@layer9/conv1/wn_g*
_output_shapes
: 
?
4layer9/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *&
_class
loc:@layer9/conv1/kernel*
dtype0*
_output_shapes
:
?
2layer9/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*&
_class
loc:@layer9/conv1/kernel*
dtype0*
_output_shapes
: 
?
2layer9/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*&
_class
loc:@layer9/conv1/kernel*
dtype0*
_output_shapes
: 
?
<layer9/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4layer9/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*&
_class
loc:@layer9/conv1/kernel
?
2layer9/conv1/kernel/Initializer/random_uniform/subSub2layer9/conv1/kernel/Initializer/random_uniform/max2layer9/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer9/conv1/kernel*
_output_shapes
: 
?
2layer9/conv1/kernel/Initializer/random_uniform/mulMul<layer9/conv1/kernel/Initializer/random_uniform/RandomUniform2layer9/conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@layer9/conv1/kernel*'
_output_shapes
:? 
?
.layer9/conv1/kernel/Initializer/random_uniformAdd2layer9/conv1/kernel/Initializer/random_uniform/mul2layer9/conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@layer9/conv1/kernel*'
_output_shapes
:? 
?
layer9/conv1/kernel
VariableV2*
shape:? *
dtype0*'
_output_shapes
:? *&
_class
loc:@layer9/conv1/kernel
?
layer9/conv1/kernel/AssignAssignlayer9/conv1/kernel.layer9/conv1/kernel/Initializer/random_uniform*
T0*&
_class
loc:@layer9/conv1/kernel*'
_output_shapes
:? 
?
layer9/conv1/kernel/readIdentitylayer9/conv1/kernel*
T0*&
_class
loc:@layer9/conv1/kernel*'
_output_shapes
:? 
?
#layer9/conv1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *$
_class
loc:@layer9/conv1/bias

layer9/conv1/bias
VariableV2*$
_class
loc:@layer9/conv1/bias*
shape: *
dtype0*
_output_shapes
: 
?
layer9/conv1/bias/AssignAssignlayer9/conv1/bias#layer9/conv1/bias/Initializer/zeros*
_output_shapes
: *
T0*$
_class
loc:@layer9/conv1/bias
?
layer9/conv1/bias/readIdentitylayer9/conv1/bias*
T0*$
_class
loc:@layer9/conv1/bias*
_output_shapes
: 
k
layer9/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
i
layer9/conv1/SquareSquarelayer9/conv1/kernel/read*'
_output_shapes
:? *
T0
w
"layer9/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
u
layer9/conv1/SumSumlayer9/conv1/Square"layer9/conv1/Sum/reduction_indices*
T0*
_output_shapes
: 
R
layer9/conv1/RsqrtRsqrtlayer9/conv1/Sum*
T0*
_output_shapes
: 
h
layer9/conv1/mulMullayer9/conv1/Rsqrtlayer9/conv1/wn_g/read*
T0*
_output_shapes
: 
w
layer9/conv1/mul_1Mullayer9/conv1/kernel/readlayer9/conv1/mul*
T0*'
_output_shapes
:? 
?
layer9/conv1/Conv2DConv2Dlayer9/Relulayer9/conv1/mul_1*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? *
T0*
strides

?
layer9/conv1/BiasAddBiasAddlayer9/conv1/Conv2Dlayer9/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 


layer9/addAddlayer9/conv1/BiasAdd
layer8/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
#layer10/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*%
_class
loc:@layer10/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer10/conv0/wn_g
VariableV2*
dtype0*
_output_shapes	
:?*%
_class
loc:@layer10/conv0/wn_g*
shape:?
?
layer10/conv0/wn_g/AssignAssignlayer10/conv0/wn_g#layer10/conv0/wn_g/Initializer/ones*
T0*%
_class
loc:@layer10/conv0/wn_g*
_output_shapes	
:?
?
layer10/conv0/wn_g/readIdentitylayer10/conv0/wn_g*
T0*%
_class
loc:@layer10/conv0/wn_g*
_output_shapes	
:?
?
5layer10/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *'
_class
loc:@layer10/conv0/kernel*
dtype0*
_output_shapes
:
?
3layer10/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*'
_class
loc:@layer10/conv0/kernel*
dtype0*
_output_shapes
: 
?
3layer10/conv0/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *?2?=*'
_class
loc:@layer10/conv0/kernel
?
=layer10/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer10/conv0/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@layer10/conv0/kernel*
dtype0*'
_output_shapes
: ?
?
3layer10/conv0/kernel/Initializer/random_uniform/subSub3layer10/conv0/kernel/Initializer/random_uniform/max3layer10/conv0/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer10/conv0/kernel*
_output_shapes
: 
?
3layer10/conv0/kernel/Initializer/random_uniform/mulMul=layer10/conv0/kernel/Initializer/random_uniform/RandomUniform3layer10/conv0/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@layer10/conv0/kernel*'
_output_shapes
: ?
?
/layer10/conv0/kernel/Initializer/random_uniformAdd3layer10/conv0/kernel/Initializer/random_uniform/mul3layer10/conv0/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer10/conv0/kernel*'
_output_shapes
: ?
?
layer10/conv0/kernel
VariableV2*'
_class
loc:@layer10/conv0/kernel*
shape: ?*
dtype0*'
_output_shapes
: ?
?
layer10/conv0/kernel/AssignAssignlayer10/conv0/kernel/layer10/conv0/kernel/Initializer/random_uniform*
T0*'
_class
loc:@layer10/conv0/kernel*'
_output_shapes
: ?
?
layer10/conv0/kernel/readIdentitylayer10/conv0/kernel*'
_output_shapes
: ?*
T0*'
_class
loc:@layer10/conv0/kernel
?
$layer10/conv0/bias/Initializer/zerosConst*
valueB?*    *%
_class
loc:@layer10/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer10/conv0/bias
VariableV2*%
_class
loc:@layer10/conv0/bias*
shape:?*
dtype0*
_output_shapes	
:?
?
layer10/conv0/bias/AssignAssignlayer10/conv0/bias$layer10/conv0/bias/Initializer/zeros*
T0*%
_class
loc:@layer10/conv0/bias*
_output_shapes	
:?
?
layer10/conv0/bias/readIdentitylayer10/conv0/bias*
T0*%
_class
loc:@layer10/conv0/bias*
_output_shapes	
:?
l
layer10/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
k
layer10/conv0/SquareSquarelayer10/conv0/kernel/read*
T0*'
_output_shapes
: ?
x
#layer10/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
y
layer10/conv0/SumSumlayer10/conv0/Square#layer10/conv0/Sum/reduction_indices*
T0*
_output_shapes	
:?
U
layer10/conv0/RsqrtRsqrtlayer10/conv0/Sum*
_output_shapes	
:?*
T0
l
layer10/conv0/mulMullayer10/conv0/Rsqrtlayer10/conv0/wn_g/read*
T0*
_output_shapes	
:?
z
layer10/conv0/mul_1Mullayer10/conv0/kernel/readlayer10/conv0/mul*
T0*'
_output_shapes
: ?
?
layer10/conv0/Conv2DConv2D
layer9/addlayer10/conv0/mul_1*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,????????????????????????????
?
layer10/conv0/BiasAddBiasAddlayer10/conv0/Conv2Dlayer10/conv0/bias/read*
T0*B
_output_shapes0
.:,????????????????????????????
x
layer10/ReluRelulayer10/conv0/BiasAdd*B
_output_shapes0
.:,????????????????????????????*
T0
?
#layer10/conv1/wn_g/Initializer/onesConst*
dtype0*
_output_shapes
: *
valueB *  ??*%
_class
loc:@layer10/conv1/wn_g
?
layer10/conv1/wn_g
VariableV2*
shape: *
dtype0*
_output_shapes
: *%
_class
loc:@layer10/conv1/wn_g
?
layer10/conv1/wn_g/AssignAssignlayer10/conv1/wn_g#layer10/conv1/wn_g/Initializer/ones*
_output_shapes
: *
T0*%
_class
loc:@layer10/conv1/wn_g
?
layer10/conv1/wn_g/readIdentitylayer10/conv1/wn_g*
T0*%
_class
loc:@layer10/conv1/wn_g*
_output_shapes
: 
?
5layer10/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *'
_class
loc:@layer10/conv1/kernel*
dtype0*
_output_shapes
:
?
3layer10/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*'
_class
loc:@layer10/conv1/kernel*
dtype0*
_output_shapes
: 
?
3layer10/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*'
_class
loc:@layer10/conv1/kernel*
dtype0*
_output_shapes
: 
?
=layer10/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer10/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*'
_class
loc:@layer10/conv1/kernel
?
3layer10/conv1/kernel/Initializer/random_uniform/subSub3layer10/conv1/kernel/Initializer/random_uniform/max3layer10/conv1/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer10/conv1/kernel*
_output_shapes
: 
?
3layer10/conv1/kernel/Initializer/random_uniform/mulMul=layer10/conv1/kernel/Initializer/random_uniform/RandomUniform3layer10/conv1/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@layer10/conv1/kernel*'
_output_shapes
:? 
?
/layer10/conv1/kernel/Initializer/random_uniformAdd3layer10/conv1/kernel/Initializer/random_uniform/mul3layer10/conv1/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer10/conv1/kernel*'
_output_shapes
:? 
?
layer10/conv1/kernel
VariableV2*'
_class
loc:@layer10/conv1/kernel*
shape:? *
dtype0*'
_output_shapes
:? 
?
layer10/conv1/kernel/AssignAssignlayer10/conv1/kernel/layer10/conv1/kernel/Initializer/random_uniform*'
_output_shapes
:? *
T0*'
_class
loc:@layer10/conv1/kernel
?
layer10/conv1/kernel/readIdentitylayer10/conv1/kernel*'
_output_shapes
:? *
T0*'
_class
loc:@layer10/conv1/kernel
?
$layer10/conv1/bias/Initializer/zerosConst*
valueB *    *%
_class
loc:@layer10/conv1/bias*
dtype0*
_output_shapes
: 
?
layer10/conv1/bias
VariableV2*
dtype0*
_output_shapes
: *%
_class
loc:@layer10/conv1/bias*
shape: 
?
layer10/conv1/bias/AssignAssignlayer10/conv1/bias$layer10/conv1/bias/Initializer/zeros*
T0*%
_class
loc:@layer10/conv1/bias*
_output_shapes
: 
?
layer10/conv1/bias/readIdentitylayer10/conv1/bias*
T0*%
_class
loc:@layer10/conv1/bias*
_output_shapes
: 
l
layer10/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
k
layer10/conv1/SquareSquarelayer10/conv1/kernel/read*'
_output_shapes
:? *
T0
x
#layer10/conv1/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
x
layer10/conv1/SumSumlayer10/conv1/Square#layer10/conv1/Sum/reduction_indices*
T0*
_output_shapes
: 
T
layer10/conv1/RsqrtRsqrtlayer10/conv1/Sum*
T0*
_output_shapes
: 
k
layer10/conv1/mulMullayer10/conv1/Rsqrtlayer10/conv1/wn_g/read*
T0*
_output_shapes
: 
z
layer10/conv1/mul_1Mullayer10/conv1/kernel/readlayer10/conv1/mul*
T0*'
_output_shapes
:? 
?
layer10/conv1/Conv2DConv2Dlayer10/Relulayer10/conv1/mul_1*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? 
?
layer10/conv1/BiasAddBiasAddlayer10/conv1/Conv2Dlayer10/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
layer10/addAddlayer10/conv1/BiasAdd
layer9/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
#layer11/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*%
_class
loc:@layer11/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer11/conv0/wn_g
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*%
_class
loc:@layer11/conv0/wn_g
?
layer11/conv0/wn_g/AssignAssignlayer11/conv0/wn_g#layer11/conv0/wn_g/Initializer/ones*
_output_shapes	
:?*
T0*%
_class
loc:@layer11/conv0/wn_g
?
layer11/conv0/wn_g/readIdentitylayer11/conv0/wn_g*
T0*%
_class
loc:@layer11/conv0/wn_g*
_output_shapes	
:?
?
5layer11/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *'
_class
loc:@layer11/conv0/kernel*
dtype0*
_output_shapes
:
?
3layer11/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*'
_class
loc:@layer11/conv0/kernel*
dtype0*
_output_shapes
: 
?
3layer11/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*'
_class
loc:@layer11/conv0/kernel*
dtype0*
_output_shapes
: 
?
=layer11/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer11/conv0/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
: ?*
T0*'
_class
loc:@layer11/conv0/kernel
?
3layer11/conv0/kernel/Initializer/random_uniform/subSub3layer11/conv0/kernel/Initializer/random_uniform/max3layer11/conv0/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@layer11/conv0/kernel
?
3layer11/conv0/kernel/Initializer/random_uniform/mulMul=layer11/conv0/kernel/Initializer/random_uniform/RandomUniform3layer11/conv0/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@layer11/conv0/kernel*'
_output_shapes
: ?
?
/layer11/conv0/kernel/Initializer/random_uniformAdd3layer11/conv0/kernel/Initializer/random_uniform/mul3layer11/conv0/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer11/conv0/kernel*'
_output_shapes
: ?
?
layer11/conv0/kernel
VariableV2*
shape: ?*
dtype0*'
_output_shapes
: ?*'
_class
loc:@layer11/conv0/kernel
?
layer11/conv0/kernel/AssignAssignlayer11/conv0/kernel/layer11/conv0/kernel/Initializer/random_uniform*'
_output_shapes
: ?*
T0*'
_class
loc:@layer11/conv0/kernel
?
layer11/conv0/kernel/readIdentitylayer11/conv0/kernel*
T0*'
_class
loc:@layer11/conv0/kernel*'
_output_shapes
: ?
?
$layer11/conv0/bias/Initializer/zerosConst*
valueB?*    *%
_class
loc:@layer11/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer11/conv0/bias
VariableV2*%
_class
loc:@layer11/conv0/bias*
shape:?*
dtype0*
_output_shapes	
:?
?
layer11/conv0/bias/AssignAssignlayer11/conv0/bias$layer11/conv0/bias/Initializer/zeros*
_output_shapes	
:?*
T0*%
_class
loc:@layer11/conv0/bias
?
layer11/conv0/bias/readIdentitylayer11/conv0/bias*
T0*%
_class
loc:@layer11/conv0/bias*
_output_shapes	
:?
l
layer11/conv0/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
k
layer11/conv0/SquareSquarelayer11/conv0/kernel/read*
T0*'
_output_shapes
: ?
x
#layer11/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
y
layer11/conv0/SumSumlayer11/conv0/Square#layer11/conv0/Sum/reduction_indices*
T0*
_output_shapes	
:?
U
layer11/conv0/RsqrtRsqrtlayer11/conv0/Sum*
T0*
_output_shapes	
:?
l
layer11/conv0/mulMullayer11/conv0/Rsqrtlayer11/conv0/wn_g/read*
T0*
_output_shapes	
:?
z
layer11/conv0/mul_1Mullayer11/conv0/kernel/readlayer11/conv0/mul*
T0*'
_output_shapes
: ?
?
layer11/conv0/Conv2DConv2Dlayer10/addlayer11/conv0/mul_1*B
_output_shapes0
.:,????????????????????????????*
T0*
strides
*
paddingSAME
?
layer11/conv0/BiasAddBiasAddlayer11/conv0/Conv2Dlayer11/conv0/bias/read*B
_output_shapes0
.:,????????????????????????????*
T0
x
layer11/ReluRelulayer11/conv0/BiasAdd*B
_output_shapes0
.:,????????????????????????????*
T0
?
#layer11/conv1/wn_g/Initializer/onesConst*
dtype0*
_output_shapes
: *
valueB *  ??*%
_class
loc:@layer11/conv1/wn_g
?
layer11/conv1/wn_g
VariableV2*
dtype0*
_output_shapes
: *%
_class
loc:@layer11/conv1/wn_g*
shape: 
?
layer11/conv1/wn_g/AssignAssignlayer11/conv1/wn_g#layer11/conv1/wn_g/Initializer/ones*
T0*%
_class
loc:@layer11/conv1/wn_g*
_output_shapes
: 
?
layer11/conv1/wn_g/readIdentitylayer11/conv1/wn_g*
T0*%
_class
loc:@layer11/conv1/wn_g*
_output_shapes
: 
?
5layer11/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *'
_class
loc:@layer11/conv1/kernel*
dtype0*
_output_shapes
:
?
3layer11/conv1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?2??*'
_class
loc:@layer11/conv1/kernel
?
3layer11/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*'
_class
loc:@layer11/conv1/kernel*
dtype0*
_output_shapes
: 
?
=layer11/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer11/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*'
_class
loc:@layer11/conv1/kernel
?
3layer11/conv1/kernel/Initializer/random_uniform/subSub3layer11/conv1/kernel/Initializer/random_uniform/max3layer11/conv1/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer11/conv1/kernel*
_output_shapes
: 
?
3layer11/conv1/kernel/Initializer/random_uniform/mulMul=layer11/conv1/kernel/Initializer/random_uniform/RandomUniform3layer11/conv1/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@layer11/conv1/kernel*'
_output_shapes
:? 
?
/layer11/conv1/kernel/Initializer/random_uniformAdd3layer11/conv1/kernel/Initializer/random_uniform/mul3layer11/conv1/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer11/conv1/kernel*'
_output_shapes
:? 
?
layer11/conv1/kernel
VariableV2*
dtype0*'
_output_shapes
:? *'
_class
loc:@layer11/conv1/kernel*
shape:? 
?
layer11/conv1/kernel/AssignAssignlayer11/conv1/kernel/layer11/conv1/kernel/Initializer/random_uniform*
T0*'
_class
loc:@layer11/conv1/kernel*'
_output_shapes
:? 
?
layer11/conv1/kernel/readIdentitylayer11/conv1/kernel*
T0*'
_class
loc:@layer11/conv1/kernel*'
_output_shapes
:? 
?
$layer11/conv1/bias/Initializer/zerosConst*
valueB *    *%
_class
loc:@layer11/conv1/bias*
dtype0*
_output_shapes
: 
?
layer11/conv1/bias
VariableV2*
dtype0*
_output_shapes
: *%
_class
loc:@layer11/conv1/bias*
shape: 
?
layer11/conv1/bias/AssignAssignlayer11/conv1/bias$layer11/conv1/bias/Initializer/zeros*
T0*%
_class
loc:@layer11/conv1/bias*
_output_shapes
: 
?
layer11/conv1/bias/readIdentitylayer11/conv1/bias*
_output_shapes
: *
T0*%
_class
loc:@layer11/conv1/bias
l
layer11/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
k
layer11/conv1/SquareSquarelayer11/conv1/kernel/read*
T0*'
_output_shapes
:? 
x
#layer11/conv1/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
x
layer11/conv1/SumSumlayer11/conv1/Square#layer11/conv1/Sum/reduction_indices*
T0*
_output_shapes
: 
T
layer11/conv1/RsqrtRsqrtlayer11/conv1/Sum*
_output_shapes
: *
T0
k
layer11/conv1/mulMullayer11/conv1/Rsqrtlayer11/conv1/wn_g/read*
T0*
_output_shapes
: 
z
layer11/conv1/mul_1Mullayer11/conv1/kernel/readlayer11/conv1/mul*
T0*'
_output_shapes
:? 
?
layer11/conv1/Conv2DConv2Dlayer11/Relulayer11/conv1/mul_1*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? 
?
layer11/conv1/BiasAddBiasAddlayer11/conv1/Conv2Dlayer11/conv1/bias/read*A
_output_shapes/
-:+??????????????????????????? *
T0
?
layer11/addAddlayer11/conv1/BiasAddlayer10/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
#layer12/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*%
_class
loc:@layer12/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer12/conv0/wn_g
VariableV2*
dtype0*
_output_shapes	
:?*%
_class
loc:@layer12/conv0/wn_g*
shape:?
?
layer12/conv0/wn_g/AssignAssignlayer12/conv0/wn_g#layer12/conv0/wn_g/Initializer/ones*
_output_shapes	
:?*
T0*%
_class
loc:@layer12/conv0/wn_g
?
layer12/conv0/wn_g/readIdentitylayer12/conv0/wn_g*
T0*%
_class
loc:@layer12/conv0/wn_g*
_output_shapes	
:?
?
5layer12/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *'
_class
loc:@layer12/conv0/kernel*
dtype0*
_output_shapes
:
?
3layer12/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*'
_class
loc:@layer12/conv0/kernel*
dtype0*
_output_shapes
: 
?
3layer12/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*'
_class
loc:@layer12/conv0/kernel*
dtype0*
_output_shapes
: 
?
=layer12/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer12/conv0/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@layer12/conv0/kernel*
dtype0*'
_output_shapes
: ?
?
3layer12/conv0/kernel/Initializer/random_uniform/subSub3layer12/conv0/kernel/Initializer/random_uniform/max3layer12/conv0/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer12/conv0/kernel*
_output_shapes
: 
?
3layer12/conv0/kernel/Initializer/random_uniform/mulMul=layer12/conv0/kernel/Initializer/random_uniform/RandomUniform3layer12/conv0/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@layer12/conv0/kernel*'
_output_shapes
: ?
?
/layer12/conv0/kernel/Initializer/random_uniformAdd3layer12/conv0/kernel/Initializer/random_uniform/mul3layer12/conv0/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer12/conv0/kernel*'
_output_shapes
: ?
?
layer12/conv0/kernel
VariableV2*
dtype0*'
_output_shapes
: ?*'
_class
loc:@layer12/conv0/kernel*
shape: ?
?
layer12/conv0/kernel/AssignAssignlayer12/conv0/kernel/layer12/conv0/kernel/Initializer/random_uniform*'
_output_shapes
: ?*
T0*'
_class
loc:@layer12/conv0/kernel
?
layer12/conv0/kernel/readIdentitylayer12/conv0/kernel*
T0*'
_class
loc:@layer12/conv0/kernel*'
_output_shapes
: ?
?
$layer12/conv0/bias/Initializer/zerosConst*
valueB?*    *%
_class
loc:@layer12/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer12/conv0/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*%
_class
loc:@layer12/conv0/bias
?
layer12/conv0/bias/AssignAssignlayer12/conv0/bias$layer12/conv0/bias/Initializer/zeros*
_output_shapes	
:?*
T0*%
_class
loc:@layer12/conv0/bias
?
layer12/conv0/bias/readIdentitylayer12/conv0/bias*
T0*%
_class
loc:@layer12/conv0/bias*
_output_shapes	
:?
l
layer12/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
k
layer12/conv0/SquareSquarelayer12/conv0/kernel/read*
T0*'
_output_shapes
: ?
x
#layer12/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
y
layer12/conv0/SumSumlayer12/conv0/Square#layer12/conv0/Sum/reduction_indices*
_output_shapes	
:?*
T0
U
layer12/conv0/RsqrtRsqrtlayer12/conv0/Sum*
T0*
_output_shapes	
:?
l
layer12/conv0/mulMullayer12/conv0/Rsqrtlayer12/conv0/wn_g/read*
T0*
_output_shapes	
:?
z
layer12/conv0/mul_1Mullayer12/conv0/kernel/readlayer12/conv0/mul*'
_output_shapes
: ?*
T0
?
layer12/conv0/Conv2DConv2Dlayer11/addlayer12/conv0/mul_1*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,????????????????????????????
?
layer12/conv0/BiasAddBiasAddlayer12/conv0/Conv2Dlayer12/conv0/bias/read*
T0*B
_output_shapes0
.:,????????????????????????????
x
layer12/ReluRelulayer12/conv0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
#layer12/conv1/wn_g/Initializer/onesConst*
dtype0*
_output_shapes
: *
valueB *  ??*%
_class
loc:@layer12/conv1/wn_g
?
layer12/conv1/wn_g
VariableV2*
dtype0*
_output_shapes
: *%
_class
loc:@layer12/conv1/wn_g*
shape: 
?
layer12/conv1/wn_g/AssignAssignlayer12/conv1/wn_g#layer12/conv1/wn_g/Initializer/ones*
_output_shapes
: *
T0*%
_class
loc:@layer12/conv1/wn_g
?
layer12/conv1/wn_g/readIdentitylayer12/conv1/wn_g*
T0*%
_class
loc:@layer12/conv1/wn_g*
_output_shapes
: 
?
5layer12/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *'
_class
loc:@layer12/conv1/kernel*
dtype0*
_output_shapes
:
?
3layer12/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*'
_class
loc:@layer12/conv1/kernel*
dtype0*
_output_shapes
: 
?
3layer12/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*'
_class
loc:@layer12/conv1/kernel*
dtype0*
_output_shapes
: 
?
=layer12/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer12/conv1/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@layer12/conv1/kernel*
dtype0*'
_output_shapes
:? 
?
3layer12/conv1/kernel/Initializer/random_uniform/subSub3layer12/conv1/kernel/Initializer/random_uniform/max3layer12/conv1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@layer12/conv1/kernel
?
3layer12/conv1/kernel/Initializer/random_uniform/mulMul=layer12/conv1/kernel/Initializer/random_uniform/RandomUniform3layer12/conv1/kernel/Initializer/random_uniform/sub*'
_output_shapes
:? *
T0*'
_class
loc:@layer12/conv1/kernel
?
/layer12/conv1/kernel/Initializer/random_uniformAdd3layer12/conv1/kernel/Initializer/random_uniform/mul3layer12/conv1/kernel/Initializer/random_uniform/min*'
_output_shapes
:? *
T0*'
_class
loc:@layer12/conv1/kernel
?
layer12/conv1/kernel
VariableV2*'
_class
loc:@layer12/conv1/kernel*
shape:? *
dtype0*'
_output_shapes
:? 
?
layer12/conv1/kernel/AssignAssignlayer12/conv1/kernel/layer12/conv1/kernel/Initializer/random_uniform*
T0*'
_class
loc:@layer12/conv1/kernel*'
_output_shapes
:? 
?
layer12/conv1/kernel/readIdentitylayer12/conv1/kernel*
T0*'
_class
loc:@layer12/conv1/kernel*'
_output_shapes
:? 
?
$layer12/conv1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *%
_class
loc:@layer12/conv1/bias
?
layer12/conv1/bias
VariableV2*
shape: *
dtype0*
_output_shapes
: *%
_class
loc:@layer12/conv1/bias
?
layer12/conv1/bias/AssignAssignlayer12/conv1/bias$layer12/conv1/bias/Initializer/zeros*
T0*%
_class
loc:@layer12/conv1/bias*
_output_shapes
: 
?
layer12/conv1/bias/readIdentitylayer12/conv1/bias*
_output_shapes
: *
T0*%
_class
loc:@layer12/conv1/bias
l
layer12/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
k
layer12/conv1/SquareSquarelayer12/conv1/kernel/read*
T0*'
_output_shapes
:? 
x
#layer12/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
x
layer12/conv1/SumSumlayer12/conv1/Square#layer12/conv1/Sum/reduction_indices*
_output_shapes
: *
T0
T
layer12/conv1/RsqrtRsqrtlayer12/conv1/Sum*
_output_shapes
: *
T0
k
layer12/conv1/mulMullayer12/conv1/Rsqrtlayer12/conv1/wn_g/read*
T0*
_output_shapes
: 
z
layer12/conv1/mul_1Mullayer12/conv1/kernel/readlayer12/conv1/mul*
T0*'
_output_shapes
:? 
?
layer12/conv1/Conv2DConv2Dlayer12/Relulayer12/conv1/mul_1*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? *
T0*
strides

?
layer12/conv1/BiasAddBiasAddlayer12/conv1/Conv2Dlayer12/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
layer12/addAddlayer12/conv1/BiasAddlayer11/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
#layer13/conv0/wn_g/Initializer/onesConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??*%
_class
loc:@layer13/conv0/wn_g
?
layer13/conv0/wn_g
VariableV2*%
_class
loc:@layer13/conv0/wn_g*
shape:?*
dtype0*
_output_shapes	
:?
?
layer13/conv0/wn_g/AssignAssignlayer13/conv0/wn_g#layer13/conv0/wn_g/Initializer/ones*
T0*%
_class
loc:@layer13/conv0/wn_g*
_output_shapes	
:?
?
layer13/conv0/wn_g/readIdentitylayer13/conv0/wn_g*
_output_shapes	
:?*
T0*%
_class
loc:@layer13/conv0/wn_g
?
5layer13/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *'
_class
loc:@layer13/conv0/kernel*
dtype0*
_output_shapes
:
?
3layer13/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*'
_class
loc:@layer13/conv0/kernel*
dtype0*
_output_shapes
: 
?
3layer13/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*'
_class
loc:@layer13/conv0/kernel*
dtype0*
_output_shapes
: 
?
=layer13/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer13/conv0/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
: ?*
T0*'
_class
loc:@layer13/conv0/kernel
?
3layer13/conv0/kernel/Initializer/random_uniform/subSub3layer13/conv0/kernel/Initializer/random_uniform/max3layer13/conv0/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer13/conv0/kernel*
_output_shapes
: 
?
3layer13/conv0/kernel/Initializer/random_uniform/mulMul=layer13/conv0/kernel/Initializer/random_uniform/RandomUniform3layer13/conv0/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@layer13/conv0/kernel*'
_output_shapes
: ?
?
/layer13/conv0/kernel/Initializer/random_uniformAdd3layer13/conv0/kernel/Initializer/random_uniform/mul3layer13/conv0/kernel/Initializer/random_uniform/min*'
_output_shapes
: ?*
T0*'
_class
loc:@layer13/conv0/kernel
?
layer13/conv0/kernel
VariableV2*'
_class
loc:@layer13/conv0/kernel*
shape: ?*
dtype0*'
_output_shapes
: ?
?
layer13/conv0/kernel/AssignAssignlayer13/conv0/kernel/layer13/conv0/kernel/Initializer/random_uniform*
T0*'
_class
loc:@layer13/conv0/kernel*'
_output_shapes
: ?
?
layer13/conv0/kernel/readIdentitylayer13/conv0/kernel*
T0*'
_class
loc:@layer13/conv0/kernel*'
_output_shapes
: ?
?
$layer13/conv0/bias/Initializer/zerosConst*
valueB?*    *%
_class
loc:@layer13/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer13/conv0/bias
VariableV2*%
_class
loc:@layer13/conv0/bias*
shape:?*
dtype0*
_output_shapes	
:?
?
layer13/conv0/bias/AssignAssignlayer13/conv0/bias$layer13/conv0/bias/Initializer/zeros*
T0*%
_class
loc:@layer13/conv0/bias*
_output_shapes	
:?
?
layer13/conv0/bias/readIdentitylayer13/conv0/bias*
_output_shapes	
:?*
T0*%
_class
loc:@layer13/conv0/bias
l
layer13/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
k
layer13/conv0/SquareSquarelayer13/conv0/kernel/read*'
_output_shapes
: ?*
T0
x
#layer13/conv0/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
y
layer13/conv0/SumSumlayer13/conv0/Square#layer13/conv0/Sum/reduction_indices*
T0*
_output_shapes	
:?
U
layer13/conv0/RsqrtRsqrtlayer13/conv0/Sum*
T0*
_output_shapes	
:?
l
layer13/conv0/mulMullayer13/conv0/Rsqrtlayer13/conv0/wn_g/read*
T0*
_output_shapes	
:?
z
layer13/conv0/mul_1Mullayer13/conv0/kernel/readlayer13/conv0/mul*
T0*'
_output_shapes
: ?
?
layer13/conv0/Conv2DConv2Dlayer12/addlayer13/conv0/mul_1*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,????????????????????????????
?
layer13/conv0/BiasAddBiasAddlayer13/conv0/Conv2Dlayer13/conv0/bias/read*B
_output_shapes0
.:,????????????????????????????*
T0
x
layer13/ReluRelulayer13/conv0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
#layer13/conv1/wn_g/Initializer/onesConst*
valueB *  ??*%
_class
loc:@layer13/conv1/wn_g*
dtype0*
_output_shapes
: 
?
layer13/conv1/wn_g
VariableV2*
shape: *
dtype0*
_output_shapes
: *%
_class
loc:@layer13/conv1/wn_g
?
layer13/conv1/wn_g/AssignAssignlayer13/conv1/wn_g#layer13/conv1/wn_g/Initializer/ones*
T0*%
_class
loc:@layer13/conv1/wn_g*
_output_shapes
: 
?
layer13/conv1/wn_g/readIdentitylayer13/conv1/wn_g*
T0*%
_class
loc:@layer13/conv1/wn_g*
_output_shapes
: 
?
5layer13/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *'
_class
loc:@layer13/conv1/kernel*
dtype0*
_output_shapes
:
?
3layer13/conv1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?2??*'
_class
loc:@layer13/conv1/kernel
?
3layer13/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*'
_class
loc:@layer13/conv1/kernel*
dtype0*
_output_shapes
: 
?
=layer13/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer13/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*'
_class
loc:@layer13/conv1/kernel
?
3layer13/conv1/kernel/Initializer/random_uniform/subSub3layer13/conv1/kernel/Initializer/random_uniform/max3layer13/conv1/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer13/conv1/kernel*
_output_shapes
: 
?
3layer13/conv1/kernel/Initializer/random_uniform/mulMul=layer13/conv1/kernel/Initializer/random_uniform/RandomUniform3layer13/conv1/kernel/Initializer/random_uniform/sub*'
_output_shapes
:? *
T0*'
_class
loc:@layer13/conv1/kernel
?
/layer13/conv1/kernel/Initializer/random_uniformAdd3layer13/conv1/kernel/Initializer/random_uniform/mul3layer13/conv1/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer13/conv1/kernel*'
_output_shapes
:? 
?
layer13/conv1/kernel
VariableV2*
dtype0*'
_output_shapes
:? *'
_class
loc:@layer13/conv1/kernel*
shape:? 
?
layer13/conv1/kernel/AssignAssignlayer13/conv1/kernel/layer13/conv1/kernel/Initializer/random_uniform*
T0*'
_class
loc:@layer13/conv1/kernel*'
_output_shapes
:? 
?
layer13/conv1/kernel/readIdentitylayer13/conv1/kernel*
T0*'
_class
loc:@layer13/conv1/kernel*'
_output_shapes
:? 
?
$layer13/conv1/bias/Initializer/zerosConst*
valueB *    *%
_class
loc:@layer13/conv1/bias*
dtype0*
_output_shapes
: 
?
layer13/conv1/bias
VariableV2*
dtype0*
_output_shapes
: *%
_class
loc:@layer13/conv1/bias*
shape: 
?
layer13/conv1/bias/AssignAssignlayer13/conv1/bias$layer13/conv1/bias/Initializer/zeros*
T0*%
_class
loc:@layer13/conv1/bias*
_output_shapes
: 
?
layer13/conv1/bias/readIdentitylayer13/conv1/bias*
_output_shapes
: *
T0*%
_class
loc:@layer13/conv1/bias
l
layer13/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
k
layer13/conv1/SquareSquarelayer13/conv1/kernel/read*
T0*'
_output_shapes
:? 
x
#layer13/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
x
layer13/conv1/SumSumlayer13/conv1/Square#layer13/conv1/Sum/reduction_indices*
T0*
_output_shapes
: 
T
layer13/conv1/RsqrtRsqrtlayer13/conv1/Sum*
T0*
_output_shapes
: 
k
layer13/conv1/mulMullayer13/conv1/Rsqrtlayer13/conv1/wn_g/read*
_output_shapes
: *
T0
z
layer13/conv1/mul_1Mullayer13/conv1/kernel/readlayer13/conv1/mul*
T0*'
_output_shapes
:? 
?
layer13/conv1/Conv2DConv2Dlayer13/Relulayer13/conv1/mul_1*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? 
?
layer13/conv1/BiasAddBiasAddlayer13/conv1/Conv2Dlayer13/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
layer13/addAddlayer13/conv1/BiasAddlayer12/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
#layer14/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*%
_class
loc:@layer14/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer14/conv0/wn_g
VariableV2*
dtype0*
_output_shapes	
:?*%
_class
loc:@layer14/conv0/wn_g*
shape:?
?
layer14/conv0/wn_g/AssignAssignlayer14/conv0/wn_g#layer14/conv0/wn_g/Initializer/ones*
T0*%
_class
loc:@layer14/conv0/wn_g*
_output_shapes	
:?
?
layer14/conv0/wn_g/readIdentitylayer14/conv0/wn_g*
T0*%
_class
loc:@layer14/conv0/wn_g*
_output_shapes	
:?
?
5layer14/conv0/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"          ?   *'
_class
loc:@layer14/conv0/kernel
?
3layer14/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*'
_class
loc:@layer14/conv0/kernel*
dtype0*
_output_shapes
: 
?
3layer14/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*'
_class
loc:@layer14/conv0/kernel*
dtype0*
_output_shapes
: 
?
=layer14/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer14/conv0/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
: ?*
T0*'
_class
loc:@layer14/conv0/kernel
?
3layer14/conv0/kernel/Initializer/random_uniform/subSub3layer14/conv0/kernel/Initializer/random_uniform/max3layer14/conv0/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@layer14/conv0/kernel
?
3layer14/conv0/kernel/Initializer/random_uniform/mulMul=layer14/conv0/kernel/Initializer/random_uniform/RandomUniform3layer14/conv0/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@layer14/conv0/kernel*'
_output_shapes
: ?
?
/layer14/conv0/kernel/Initializer/random_uniformAdd3layer14/conv0/kernel/Initializer/random_uniform/mul3layer14/conv0/kernel/Initializer/random_uniform/min*'
_output_shapes
: ?*
T0*'
_class
loc:@layer14/conv0/kernel
?
layer14/conv0/kernel
VariableV2*
dtype0*'
_output_shapes
: ?*'
_class
loc:@layer14/conv0/kernel*
shape: ?
?
layer14/conv0/kernel/AssignAssignlayer14/conv0/kernel/layer14/conv0/kernel/Initializer/random_uniform*'
_output_shapes
: ?*
T0*'
_class
loc:@layer14/conv0/kernel
?
layer14/conv0/kernel/readIdentitylayer14/conv0/kernel*
T0*'
_class
loc:@layer14/conv0/kernel*'
_output_shapes
: ?
?
$layer14/conv0/bias/Initializer/zerosConst*
valueB?*    *%
_class
loc:@layer14/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer14/conv0/bias
VariableV2*
dtype0*
_output_shapes	
:?*%
_class
loc:@layer14/conv0/bias*
shape:?
?
layer14/conv0/bias/AssignAssignlayer14/conv0/bias$layer14/conv0/bias/Initializer/zeros*
T0*%
_class
loc:@layer14/conv0/bias*
_output_shapes	
:?
?
layer14/conv0/bias/readIdentitylayer14/conv0/bias*
T0*%
_class
loc:@layer14/conv0/bias*
_output_shapes	
:?
l
layer14/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
k
layer14/conv0/SquareSquarelayer14/conv0/kernel/read*
T0*'
_output_shapes
: ?
x
#layer14/conv0/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
y
layer14/conv0/SumSumlayer14/conv0/Square#layer14/conv0/Sum/reduction_indices*
_output_shapes	
:?*
T0
U
layer14/conv0/RsqrtRsqrtlayer14/conv0/Sum*
T0*
_output_shapes	
:?
l
layer14/conv0/mulMullayer14/conv0/Rsqrtlayer14/conv0/wn_g/read*
_output_shapes	
:?*
T0
z
layer14/conv0/mul_1Mullayer14/conv0/kernel/readlayer14/conv0/mul*'
_output_shapes
: ?*
T0
?
layer14/conv0/Conv2DConv2Dlayer13/addlayer14/conv0/mul_1*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,????????????????????????????
?
layer14/conv0/BiasAddBiasAddlayer14/conv0/Conv2Dlayer14/conv0/bias/read*
T0*B
_output_shapes0
.:,????????????????????????????
x
layer14/ReluRelulayer14/conv0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
#layer14/conv1/wn_g/Initializer/onesConst*
valueB *  ??*%
_class
loc:@layer14/conv1/wn_g*
dtype0*
_output_shapes
: 
?
layer14/conv1/wn_g
VariableV2*
shape: *
dtype0*
_output_shapes
: *%
_class
loc:@layer14/conv1/wn_g
?
layer14/conv1/wn_g/AssignAssignlayer14/conv1/wn_g#layer14/conv1/wn_g/Initializer/ones*
T0*%
_class
loc:@layer14/conv1/wn_g*
_output_shapes
: 
?
layer14/conv1/wn_g/readIdentitylayer14/conv1/wn_g*
T0*%
_class
loc:@layer14/conv1/wn_g*
_output_shapes
: 
?
5layer14/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?       *'
_class
loc:@layer14/conv1/kernel*
dtype0*
_output_shapes
:
?
3layer14/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*'
_class
loc:@layer14/conv1/kernel*
dtype0*
_output_shapes
: 
?
3layer14/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*'
_class
loc:@layer14/conv1/kernel*
dtype0*
_output_shapes
: 
?
=layer14/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer14/conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:? *
T0*'
_class
loc:@layer14/conv1/kernel
?
3layer14/conv1/kernel/Initializer/random_uniform/subSub3layer14/conv1/kernel/Initializer/random_uniform/max3layer14/conv1/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer14/conv1/kernel*
_output_shapes
: 
?
3layer14/conv1/kernel/Initializer/random_uniform/mulMul=layer14/conv1/kernel/Initializer/random_uniform/RandomUniform3layer14/conv1/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@layer14/conv1/kernel*'
_output_shapes
:? 
?
/layer14/conv1/kernel/Initializer/random_uniformAdd3layer14/conv1/kernel/Initializer/random_uniform/mul3layer14/conv1/kernel/Initializer/random_uniform/min*'
_output_shapes
:? *
T0*'
_class
loc:@layer14/conv1/kernel
?
layer14/conv1/kernel
VariableV2*
shape:? *
dtype0*'
_output_shapes
:? *'
_class
loc:@layer14/conv1/kernel
?
layer14/conv1/kernel/AssignAssignlayer14/conv1/kernel/layer14/conv1/kernel/Initializer/random_uniform*
T0*'
_class
loc:@layer14/conv1/kernel*'
_output_shapes
:? 
?
layer14/conv1/kernel/readIdentitylayer14/conv1/kernel*
T0*'
_class
loc:@layer14/conv1/kernel*'
_output_shapes
:? 
?
$layer14/conv1/bias/Initializer/zerosConst*
valueB *    *%
_class
loc:@layer14/conv1/bias*
dtype0*
_output_shapes
: 
?
layer14/conv1/bias
VariableV2*%
_class
loc:@layer14/conv1/bias*
shape: *
dtype0*
_output_shapes
: 
?
layer14/conv1/bias/AssignAssignlayer14/conv1/bias$layer14/conv1/bias/Initializer/zeros*
T0*%
_class
loc:@layer14/conv1/bias*
_output_shapes
: 
?
layer14/conv1/bias/readIdentitylayer14/conv1/bias*
T0*%
_class
loc:@layer14/conv1/bias*
_output_shapes
: 
l
layer14/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
k
layer14/conv1/SquareSquarelayer14/conv1/kernel/read*
T0*'
_output_shapes
:? 
x
#layer14/conv1/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
x
layer14/conv1/SumSumlayer14/conv1/Square#layer14/conv1/Sum/reduction_indices*
T0*
_output_shapes
: 
T
layer14/conv1/RsqrtRsqrtlayer14/conv1/Sum*
T0*
_output_shapes
: 
k
layer14/conv1/mulMullayer14/conv1/Rsqrtlayer14/conv1/wn_g/read*
T0*
_output_shapes
: 
z
layer14/conv1/mul_1Mullayer14/conv1/kernel/readlayer14/conv1/mul*
T0*'
_output_shapes
:? 
?
layer14/conv1/Conv2DConv2Dlayer14/Relulayer14/conv1/mul_1*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? *
T0*
strides

?
layer14/conv1/BiasAddBiasAddlayer14/conv1/Conv2Dlayer14/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
layer14/addAddlayer14/conv1/BiasAddlayer13/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
#layer15/conv0/wn_g/Initializer/onesConst*
valueB?*  ??*%
_class
loc:@layer15/conv0/wn_g*
dtype0*
_output_shapes	
:?
?
layer15/conv0/wn_g
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*%
_class
loc:@layer15/conv0/wn_g
?
layer15/conv0/wn_g/AssignAssignlayer15/conv0/wn_g#layer15/conv0/wn_g/Initializer/ones*
T0*%
_class
loc:@layer15/conv0/wn_g*
_output_shapes	
:?
?
layer15/conv0/wn_g/readIdentitylayer15/conv0/wn_g*
_output_shapes	
:?*
T0*%
_class
loc:@layer15/conv0/wn_g
?
5layer15/conv0/kernel/Initializer/random_uniform/shapeConst*%
valueB"          ?   *'
_class
loc:@layer15/conv0/kernel*
dtype0*
_output_shapes
:
?
3layer15/conv0/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*'
_class
loc:@layer15/conv0/kernel*
dtype0*
_output_shapes
: 
?
3layer15/conv0/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*'
_class
loc:@layer15/conv0/kernel*
dtype0*
_output_shapes
: 
?
=layer15/conv0/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer15/conv0/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
: ?*
T0*'
_class
loc:@layer15/conv0/kernel
?
3layer15/conv0/kernel/Initializer/random_uniform/subSub3layer15/conv0/kernel/Initializer/random_uniform/max3layer15/conv0/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer15/conv0/kernel*
_output_shapes
: 
?
3layer15/conv0/kernel/Initializer/random_uniform/mulMul=layer15/conv0/kernel/Initializer/random_uniform/RandomUniform3layer15/conv0/kernel/Initializer/random_uniform/sub*'
_output_shapes
: ?*
T0*'
_class
loc:@layer15/conv0/kernel
?
/layer15/conv0/kernel/Initializer/random_uniformAdd3layer15/conv0/kernel/Initializer/random_uniform/mul3layer15/conv0/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer15/conv0/kernel*'
_output_shapes
: ?
?
layer15/conv0/kernel
VariableV2*
dtype0*'
_output_shapes
: ?*'
_class
loc:@layer15/conv0/kernel*
shape: ?
?
layer15/conv0/kernel/AssignAssignlayer15/conv0/kernel/layer15/conv0/kernel/Initializer/random_uniform*
T0*'
_class
loc:@layer15/conv0/kernel*'
_output_shapes
: ?
?
layer15/conv0/kernel/readIdentitylayer15/conv0/kernel*
T0*'
_class
loc:@layer15/conv0/kernel*'
_output_shapes
: ?
?
$layer15/conv0/bias/Initializer/zerosConst*
valueB?*    *%
_class
loc:@layer15/conv0/bias*
dtype0*
_output_shapes	
:?
?
layer15/conv0/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*%
_class
loc:@layer15/conv0/bias
?
layer15/conv0/bias/AssignAssignlayer15/conv0/bias$layer15/conv0/bias/Initializer/zeros*
T0*%
_class
loc:@layer15/conv0/bias*
_output_shapes	
:?
?
layer15/conv0/bias/readIdentitylayer15/conv0/bias*
_output_shapes	
:?*
T0*%
_class
loc:@layer15/conv0/bias
l
layer15/conv0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
k
layer15/conv0/SquareSquarelayer15/conv0/kernel/read*
T0*'
_output_shapes
: ?
x
#layer15/conv0/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
y
layer15/conv0/SumSumlayer15/conv0/Square#layer15/conv0/Sum/reduction_indices*
T0*
_output_shapes	
:?
U
layer15/conv0/RsqrtRsqrtlayer15/conv0/Sum*
T0*
_output_shapes	
:?
l
layer15/conv0/mulMullayer15/conv0/Rsqrtlayer15/conv0/wn_g/read*
T0*
_output_shapes	
:?
z
layer15/conv0/mul_1Mullayer15/conv0/kernel/readlayer15/conv0/mul*
T0*'
_output_shapes
: ?
?
layer15/conv0/Conv2DConv2Dlayer14/addlayer15/conv0/mul_1*B
_output_shapes0
.:,????????????????????????????*
T0*
strides
*
paddingSAME
?
layer15/conv0/BiasAddBiasAddlayer15/conv0/Conv2Dlayer15/conv0/bias/read*B
_output_shapes0
.:,????????????????????????????*
T0
x
layer15/ReluRelulayer15/conv0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
#layer15/conv1/wn_g/Initializer/onesConst*
dtype0*
_output_shapes
: *
valueB *  ??*%
_class
loc:@layer15/conv1/wn_g
?
layer15/conv1/wn_g
VariableV2*
dtype0*
_output_shapes
: *%
_class
loc:@layer15/conv1/wn_g*
shape: 
?
layer15/conv1/wn_g/AssignAssignlayer15/conv1/wn_g#layer15/conv1/wn_g/Initializer/ones*
T0*%
_class
loc:@layer15/conv1/wn_g*
_output_shapes
: 
?
layer15/conv1/wn_g/readIdentitylayer15/conv1/wn_g*
T0*%
_class
loc:@layer15/conv1/wn_g*
_output_shapes
: 
?
5layer15/conv1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      ?       *'
_class
loc:@layer15/conv1/kernel
?
3layer15/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *?2??*'
_class
loc:@layer15/conv1/kernel*
dtype0*
_output_shapes
: 
?
3layer15/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?2?=*'
_class
loc:@layer15/conv1/kernel*
dtype0*
_output_shapes
: 
?
=layer15/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform5layer15/conv1/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@layer15/conv1/kernel*
dtype0*'
_output_shapes
:? 
?
3layer15/conv1/kernel/Initializer/random_uniform/subSub3layer15/conv1/kernel/Initializer/random_uniform/max3layer15/conv1/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer15/conv1/kernel*
_output_shapes
: 
?
3layer15/conv1/kernel/Initializer/random_uniform/mulMul=layer15/conv1/kernel/Initializer/random_uniform/RandomUniform3layer15/conv1/kernel/Initializer/random_uniform/sub*'
_output_shapes
:? *
T0*'
_class
loc:@layer15/conv1/kernel
?
/layer15/conv1/kernel/Initializer/random_uniformAdd3layer15/conv1/kernel/Initializer/random_uniform/mul3layer15/conv1/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@layer15/conv1/kernel*'
_output_shapes
:? 
?
layer15/conv1/kernel
VariableV2*
shape:? *
dtype0*'
_output_shapes
:? *'
_class
loc:@layer15/conv1/kernel
?
layer15/conv1/kernel/AssignAssignlayer15/conv1/kernel/layer15/conv1/kernel/Initializer/random_uniform*'
_output_shapes
:? *
T0*'
_class
loc:@layer15/conv1/kernel
?
layer15/conv1/kernel/readIdentitylayer15/conv1/kernel*'
_output_shapes
:? *
T0*'
_class
loc:@layer15/conv1/kernel
?
$layer15/conv1/bias/Initializer/zerosConst*
valueB *    *%
_class
loc:@layer15/conv1/bias*
dtype0*
_output_shapes
: 
?
layer15/conv1/bias
VariableV2*
shape: *
dtype0*
_output_shapes
: *%
_class
loc:@layer15/conv1/bias
?
layer15/conv1/bias/AssignAssignlayer15/conv1/bias$layer15/conv1/bias/Initializer/zeros*
T0*%
_class
loc:@layer15/conv1/bias*
_output_shapes
: 
?
layer15/conv1/bias/readIdentitylayer15/conv1/bias*
T0*%
_class
loc:@layer15/conv1/bias*
_output_shapes
: 
l
layer15/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
k
layer15/conv1/SquareSquarelayer15/conv1/kernel/read*
T0*'
_output_shapes
:? 
x
#layer15/conv1/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
x
layer15/conv1/SumSumlayer15/conv1/Square#layer15/conv1/Sum/reduction_indices*
_output_shapes
: *
T0
T
layer15/conv1/RsqrtRsqrtlayer15/conv1/Sum*
T0*
_output_shapes
: 
k
layer15/conv1/mulMullayer15/conv1/Rsqrtlayer15/conv1/wn_g/read*
_output_shapes
: *
T0
z
layer15/conv1/mul_1Mullayer15/conv1/kernel/readlayer15/conv1/mul*
T0*'
_output_shapes
:? 
?
layer15/conv1/Conv2DConv2Dlayer15/Relulayer15/conv1/mul_1*
paddingSAME*A
_output_shapes/
-:+??????????????????????????? *
T0*
strides

?
layer15/conv1/BiasAddBiasAddlayer15/conv1/Conv2Dlayer15/conv1/bias/read*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
layer15/addAddlayer15/conv1/BiasAddlayer14/add*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
/output/conv2d_weight_norm/wn_g/Initializer/onesConst*
valueB*  ??*1
_class'
%#loc:@output/conv2d_weight_norm/wn_g*
dtype0*
_output_shapes
:
?
output/conv2d_weight_norm/wn_g
VariableV2*
dtype0*
_output_shapes
:*1
_class'
%#loc:@output/conv2d_weight_norm/wn_g*
shape:
?
%output/conv2d_weight_norm/wn_g/AssignAssignoutput/conv2d_weight_norm/wn_g/output/conv2d_weight_norm/wn_g/Initializer/ones*
T0*1
_class'
%#loc:@output/conv2d_weight_norm/wn_g*
_output_shapes
:
?
#output/conv2d_weight_norm/wn_g/readIdentityoutput/conv2d_weight_norm/wn_g*
_output_shapes
:*
T0*1
_class'
%#loc:@output/conv2d_weight_norm/wn_g
?
Aoutput/conv2d_weight_norm/kernel/Initializer/random_uniform/shapeConst*%
valueB"             *3
_class)
'%loc:@output/conv2d_weight_norm/kernel*
dtype0*
_output_shapes
:
?
?output/conv2d_weight_norm/kernel/Initializer/random_uniform/minConst*
valueB
 *d??*3
_class)
'%loc:@output/conv2d_weight_norm/kernel*
dtype0*
_output_shapes
: 
?
?output/conv2d_weight_norm/kernel/Initializer/random_uniform/maxConst*
valueB
 *d?=*3
_class)
'%loc:@output/conv2d_weight_norm/kernel*
dtype0*
_output_shapes
: 
?
Ioutput/conv2d_weight_norm/kernel/Initializer/random_uniform/RandomUniformRandomUniformAoutput/conv2d_weight_norm/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *
T0*3
_class)
'%loc:@output/conv2d_weight_norm/kernel
?
?output/conv2d_weight_norm/kernel/Initializer/random_uniform/subSub?output/conv2d_weight_norm/kernel/Initializer/random_uniform/max?output/conv2d_weight_norm/kernel/Initializer/random_uniform/min*
T0*3
_class)
'%loc:@output/conv2d_weight_norm/kernel*
_output_shapes
: 
?
?output/conv2d_weight_norm/kernel/Initializer/random_uniform/mulMulIoutput/conv2d_weight_norm/kernel/Initializer/random_uniform/RandomUniform?output/conv2d_weight_norm/kernel/Initializer/random_uniform/sub*
T0*3
_class)
'%loc:@output/conv2d_weight_norm/kernel*&
_output_shapes
: 
?
;output/conv2d_weight_norm/kernel/Initializer/random_uniformAdd?output/conv2d_weight_norm/kernel/Initializer/random_uniform/mul?output/conv2d_weight_norm/kernel/Initializer/random_uniform/min*
T0*3
_class)
'%loc:@output/conv2d_weight_norm/kernel*&
_output_shapes
: 
?
 output/conv2d_weight_norm/kernel
VariableV2*
shape: *
dtype0*&
_output_shapes
: *3
_class)
'%loc:@output/conv2d_weight_norm/kernel
?
'output/conv2d_weight_norm/kernel/AssignAssign output/conv2d_weight_norm/kernel;output/conv2d_weight_norm/kernel/Initializer/random_uniform*
T0*3
_class)
'%loc:@output/conv2d_weight_norm/kernel*&
_output_shapes
: 
?
%output/conv2d_weight_norm/kernel/readIdentity output/conv2d_weight_norm/kernel*
T0*3
_class)
'%loc:@output/conv2d_weight_norm/kernel*&
_output_shapes
: 
?
0output/conv2d_weight_norm/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *1
_class'
%#loc:@output/conv2d_weight_norm/bias
?
output/conv2d_weight_norm/bias
VariableV2*1
_class'
%#loc:@output/conv2d_weight_norm/bias*
shape:*
dtype0*
_output_shapes
:
?
%output/conv2d_weight_norm/bias/AssignAssignoutput/conv2d_weight_norm/bias0output/conv2d_weight_norm/bias/Initializer/zeros*
T0*1
_class'
%#loc:@output/conv2d_weight_norm/bias*
_output_shapes
:
?
#output/conv2d_weight_norm/bias/readIdentityoutput/conv2d_weight_norm/bias*
T0*1
_class'
%#loc:@output/conv2d_weight_norm/bias*
_output_shapes
:
x
'output/conv2d_weight_norm/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
 output/conv2d_weight_norm/SquareSquare%output/conv2d_weight_norm/kernel/read*&
_output_shapes
: *
T0
?
/output/conv2d_weight_norm/Sum/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
?
output/conv2d_weight_norm/SumSum output/conv2d_weight_norm/Square/output/conv2d_weight_norm/Sum/reduction_indices*
T0*
_output_shapes
:
l
output/conv2d_weight_norm/RsqrtRsqrtoutput/conv2d_weight_norm/Sum*
T0*
_output_shapes
:
?
output/conv2d_weight_norm/mulMuloutput/conv2d_weight_norm/Rsqrt#output/conv2d_weight_norm/wn_g/read*
_output_shapes
:*
T0
?
output/conv2d_weight_norm/mul_1Mul%output/conv2d_weight_norm/kernel/readoutput/conv2d_weight_norm/mul*
T0*&
_output_shapes
: 
?
 output/conv2d_weight_norm/Conv2DConv2Dlayer15/addoutput/conv2d_weight_norm/mul_1*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+???????????????????????????
?
!output/conv2d_weight_norm/BiasAddBiasAdd output/conv2d_weight_norm/Conv2D#output/conv2d_weight_norm/bias/read*
T0*A
_output_shapes/
-:+???????????????????????????
?
output/DepthToSpaceDepthToSpace!output/conv2d_weight_norm/BiasAdd*

block_size*
T0*A
_output_shapes/
-:+???????????????????????????
~
addAddoutput/DepthToSpaceskip/DepthToSpace*
T0*A
_output_shapes/
-:+???????????????????????????
d
add_1AddaddConst*
T0*A
_output_shapes/
-:+???????????????????????????
\
clip_by_value/Minimum/yConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
clip_by_value/MinimumMinimumadd_1clip_by_value/Minimum/y*
T0*A
_output_shapes/
-:+???????????????????????????
T
clip_by_value/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
clip_by_valueMaximumclip_by_value/Minimumclip_by_value/y*
T0*A
_output_shapes/
-:+???????????????????????????

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
?
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_0e5319688dc0423ebd06a2dfeef99e5e/part
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?jBglobal_stepBinput/conv2d_weight_norm/biasBinput/conv2d_weight_norm/kernelBinput/conv2d_weight_norm/wn_gBlayer0/conv0/biasBlayer0/conv0/kernelBlayer0/conv0/wn_gBlayer0/conv1/biasBlayer0/conv1/kernelBlayer0/conv1/wn_gBlayer1/conv0/biasBlayer1/conv0/kernelBlayer1/conv0/wn_gBlayer1/conv1/biasBlayer1/conv1/kernelBlayer1/conv1/wn_gBlayer10/conv0/biasBlayer10/conv0/kernelBlayer10/conv0/wn_gBlayer10/conv1/biasBlayer10/conv1/kernelBlayer10/conv1/wn_gBlayer11/conv0/biasBlayer11/conv0/kernelBlayer11/conv0/wn_gBlayer11/conv1/biasBlayer11/conv1/kernelBlayer11/conv1/wn_gBlayer12/conv0/biasBlayer12/conv0/kernelBlayer12/conv0/wn_gBlayer12/conv1/biasBlayer12/conv1/kernelBlayer12/conv1/wn_gBlayer13/conv0/biasBlayer13/conv0/kernelBlayer13/conv0/wn_gBlayer13/conv1/biasBlayer13/conv1/kernelBlayer13/conv1/wn_gBlayer14/conv0/biasBlayer14/conv0/kernelBlayer14/conv0/wn_gBlayer14/conv1/biasBlayer14/conv1/kernelBlayer14/conv1/wn_gBlayer15/conv0/biasBlayer15/conv0/kernelBlayer15/conv0/wn_gBlayer15/conv1/biasBlayer15/conv1/kernelBlayer15/conv1/wn_gBlayer2/conv0/biasBlayer2/conv0/kernelBlayer2/conv0/wn_gBlayer2/conv1/biasBlayer2/conv1/kernelBlayer2/conv1/wn_gBlayer3/conv0/biasBlayer3/conv0/kernelBlayer3/conv0/wn_gBlayer3/conv1/biasBlayer3/conv1/kernelBlayer3/conv1/wn_gBlayer4/conv0/biasBlayer4/conv0/kernelBlayer4/conv0/wn_gBlayer4/conv1/biasBlayer4/conv1/kernelBlayer4/conv1/wn_gBlayer5/conv0/biasBlayer5/conv0/kernelBlayer5/conv0/wn_gBlayer5/conv1/biasBlayer5/conv1/kernelBlayer5/conv1/wn_gBlayer6/conv0/biasBlayer6/conv0/kernelBlayer6/conv0/wn_gBlayer6/conv1/biasBlayer6/conv1/kernelBlayer6/conv1/wn_gBlayer7/conv0/biasBlayer7/conv0/kernelBlayer7/conv0/wn_gBlayer7/conv1/biasBlayer7/conv1/kernelBlayer7/conv1/wn_gBlayer8/conv0/biasBlayer8/conv0/kernelBlayer8/conv0/wn_gBlayer8/conv1/biasBlayer8/conv1/kernelBlayer8/conv1/wn_gBlayer9/conv0/biasBlayer9/conv0/kernelBlayer9/conv0/wn_gBlayer9/conv1/biasBlayer9/conv1/kernelBlayer9/conv1/wn_gBoutput/conv2d_weight_norm/biasB output/conv2d_weight_norm/kernelBoutput/conv2d_weight_norm/wn_gBskip/conv2d_weight_norm/biasBskip/conv2d_weight_norm/kernelBskip/conv2d_weight_norm/wn_g*
dtype0*
_output_shapes
:j
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:j*?
value?B?jB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_stepinput/conv2d_weight_norm/biasinput/conv2d_weight_norm/kernelinput/conv2d_weight_norm/wn_glayer0/conv0/biaslayer0/conv0/kernellayer0/conv0/wn_glayer0/conv1/biaslayer0/conv1/kernellayer0/conv1/wn_glayer1/conv0/biaslayer1/conv0/kernellayer1/conv0/wn_glayer1/conv1/biaslayer1/conv1/kernellayer1/conv1/wn_glayer10/conv0/biaslayer10/conv0/kernellayer10/conv0/wn_glayer10/conv1/biaslayer10/conv1/kernellayer10/conv1/wn_glayer11/conv0/biaslayer11/conv0/kernellayer11/conv0/wn_glayer11/conv1/biaslayer11/conv1/kernellayer11/conv1/wn_glayer12/conv0/biaslayer12/conv0/kernellayer12/conv0/wn_glayer12/conv1/biaslayer12/conv1/kernellayer12/conv1/wn_glayer13/conv0/biaslayer13/conv0/kernellayer13/conv0/wn_glayer13/conv1/biaslayer13/conv1/kernellayer13/conv1/wn_glayer14/conv0/biaslayer14/conv0/kernellayer14/conv0/wn_glayer14/conv1/biaslayer14/conv1/kernellayer14/conv1/wn_glayer15/conv0/biaslayer15/conv0/kernellayer15/conv0/wn_glayer15/conv1/biaslayer15/conv1/kernellayer15/conv1/wn_glayer2/conv0/biaslayer2/conv0/kernellayer2/conv0/wn_glayer2/conv1/biaslayer2/conv1/kernellayer2/conv1/wn_glayer3/conv0/biaslayer3/conv0/kernellayer3/conv0/wn_glayer3/conv1/biaslayer3/conv1/kernellayer3/conv1/wn_glayer4/conv0/biaslayer4/conv0/kernellayer4/conv0/wn_glayer4/conv1/biaslayer4/conv1/kernellayer4/conv1/wn_glayer5/conv0/biaslayer5/conv0/kernellayer5/conv0/wn_glayer5/conv1/biaslayer5/conv1/kernellayer5/conv1/wn_glayer6/conv0/biaslayer6/conv0/kernellayer6/conv0/wn_glayer6/conv1/biaslayer6/conv1/kernellayer6/conv1/wn_glayer7/conv0/biaslayer7/conv0/kernellayer7/conv0/wn_glayer7/conv1/biaslayer7/conv1/kernellayer7/conv1/wn_glayer8/conv0/biaslayer8/conv0/kernellayer8/conv0/wn_glayer8/conv1/biaslayer8/conv1/kernellayer8/conv1/wn_glayer9/conv0/biaslayer9/conv0/kernellayer9/conv0/wn_glayer9/conv1/biaslayer9/conv1/kernellayer9/conv1/wn_goutput/conv2d_weight_norm/bias output/conv2d_weight_norm/kerneloutput/conv2d_weight_norm/wn_gskip/conv2d_weight_norm/biasskip/conv2d_weight_norm/kernelskip/conv2d_weight_norm/wn_g"/device:CPU:0*x
dtypesn
l2j	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?jBglobal_stepBinput/conv2d_weight_norm/biasBinput/conv2d_weight_norm/kernelBinput/conv2d_weight_norm/wn_gBlayer0/conv0/biasBlayer0/conv0/kernelBlayer0/conv0/wn_gBlayer0/conv1/biasBlayer0/conv1/kernelBlayer0/conv1/wn_gBlayer1/conv0/biasBlayer1/conv0/kernelBlayer1/conv0/wn_gBlayer1/conv1/biasBlayer1/conv1/kernelBlayer1/conv1/wn_gBlayer10/conv0/biasBlayer10/conv0/kernelBlayer10/conv0/wn_gBlayer10/conv1/biasBlayer10/conv1/kernelBlayer10/conv1/wn_gBlayer11/conv0/biasBlayer11/conv0/kernelBlayer11/conv0/wn_gBlayer11/conv1/biasBlayer11/conv1/kernelBlayer11/conv1/wn_gBlayer12/conv0/biasBlayer12/conv0/kernelBlayer12/conv0/wn_gBlayer12/conv1/biasBlayer12/conv1/kernelBlayer12/conv1/wn_gBlayer13/conv0/biasBlayer13/conv0/kernelBlayer13/conv0/wn_gBlayer13/conv1/biasBlayer13/conv1/kernelBlayer13/conv1/wn_gBlayer14/conv0/biasBlayer14/conv0/kernelBlayer14/conv0/wn_gBlayer14/conv1/biasBlayer14/conv1/kernelBlayer14/conv1/wn_gBlayer15/conv0/biasBlayer15/conv0/kernelBlayer15/conv0/wn_gBlayer15/conv1/biasBlayer15/conv1/kernelBlayer15/conv1/wn_gBlayer2/conv0/biasBlayer2/conv0/kernelBlayer2/conv0/wn_gBlayer2/conv1/biasBlayer2/conv1/kernelBlayer2/conv1/wn_gBlayer3/conv0/biasBlayer3/conv0/kernelBlayer3/conv0/wn_gBlayer3/conv1/biasBlayer3/conv1/kernelBlayer3/conv1/wn_gBlayer4/conv0/biasBlayer4/conv0/kernelBlayer4/conv0/wn_gBlayer4/conv1/biasBlayer4/conv1/kernelBlayer4/conv1/wn_gBlayer5/conv0/biasBlayer5/conv0/kernelBlayer5/conv0/wn_gBlayer5/conv1/biasBlayer5/conv1/kernelBlayer5/conv1/wn_gBlayer6/conv0/biasBlayer6/conv0/kernelBlayer6/conv0/wn_gBlayer6/conv1/biasBlayer6/conv1/kernelBlayer6/conv1/wn_gBlayer7/conv0/biasBlayer7/conv0/kernelBlayer7/conv0/wn_gBlayer7/conv1/biasBlayer7/conv1/kernelBlayer7/conv1/wn_gBlayer8/conv0/biasBlayer8/conv0/kernelBlayer8/conv0/wn_gBlayer8/conv1/biasBlayer8/conv1/kernelBlayer8/conv1/wn_gBlayer9/conv0/biasBlayer9/conv0/kernelBlayer9/conv0/wn_gBlayer9/conv1/biasBlayer9/conv1/kernelBlayer9/conv1/wn_gBoutput/conv2d_weight_norm/biasB output/conv2d_weight_norm/kernelBoutput/conv2d_weight_norm/wn_gBskip/conv2d_weight_norm/biasBskip/conv2d_weight_norm/kernelBskip/conv2d_weight_norm/wn_g*
dtype0*
_output_shapes
:j
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*?
value?B?jB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:j
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*x
dtypesn
l2j	
s
save/AssignAssignglobal_stepsave/RestoreV2*
T0	*
_class
loc:@global_step*
_output_shapes
: 
?
save/Assign_1Assigninput/conv2d_weight_norm/biassave/RestoreV2:1*
_output_shapes
: *
T0*0
_class&
$"loc:@input/conv2d_weight_norm/bias
?
save/Assign_2Assigninput/conv2d_weight_norm/kernelsave/RestoreV2:2*
T0*2
_class(
&$loc:@input/conv2d_weight_norm/kernel*&
_output_shapes
: 
?
save/Assign_3Assigninput/conv2d_weight_norm/wn_gsave/RestoreV2:3*
_output_shapes
: *
T0*0
_class&
$"loc:@input/conv2d_weight_norm/wn_g
?
save/Assign_4Assignlayer0/conv0/biassave/RestoreV2:4*
_output_shapes	
:?*
T0*$
_class
loc:@layer0/conv0/bias
?
save/Assign_5Assignlayer0/conv0/kernelsave/RestoreV2:5*
T0*&
_class
loc:@layer0/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_6Assignlayer0/conv0/wn_gsave/RestoreV2:6*
T0*$
_class
loc:@layer0/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_7Assignlayer0/conv1/biassave/RestoreV2:7*
_output_shapes
: *
T0*$
_class
loc:@layer0/conv1/bias
?
save/Assign_8Assignlayer0/conv1/kernelsave/RestoreV2:8*
T0*&
_class
loc:@layer0/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_9Assignlayer0/conv1/wn_gsave/RestoreV2:9*
T0*$
_class
loc:@layer0/conv1/wn_g*
_output_shapes
: 
?
save/Assign_10Assignlayer1/conv0/biassave/RestoreV2:10*
T0*$
_class
loc:@layer1/conv0/bias*
_output_shapes	
:?
?
save/Assign_11Assignlayer1/conv0/kernelsave/RestoreV2:11*'
_output_shapes
: ?*
T0*&
_class
loc:@layer1/conv0/kernel
?
save/Assign_12Assignlayer1/conv0/wn_gsave/RestoreV2:12*
T0*$
_class
loc:@layer1/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_13Assignlayer1/conv1/biassave/RestoreV2:13*
_output_shapes
: *
T0*$
_class
loc:@layer1/conv1/bias
?
save/Assign_14Assignlayer1/conv1/kernelsave/RestoreV2:14*
T0*&
_class
loc:@layer1/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_15Assignlayer1/conv1/wn_gsave/RestoreV2:15*
T0*$
_class
loc:@layer1/conv1/wn_g*
_output_shapes
: 
?
save/Assign_16Assignlayer10/conv0/biassave/RestoreV2:16*
T0*%
_class
loc:@layer10/conv0/bias*
_output_shapes	
:?
?
save/Assign_17Assignlayer10/conv0/kernelsave/RestoreV2:17*
T0*'
_class
loc:@layer10/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_18Assignlayer10/conv0/wn_gsave/RestoreV2:18*
T0*%
_class
loc:@layer10/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_19Assignlayer10/conv1/biassave/RestoreV2:19*
_output_shapes
: *
T0*%
_class
loc:@layer10/conv1/bias
?
save/Assign_20Assignlayer10/conv1/kernelsave/RestoreV2:20*
T0*'
_class
loc:@layer10/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_21Assignlayer10/conv1/wn_gsave/RestoreV2:21*
T0*%
_class
loc:@layer10/conv1/wn_g*
_output_shapes
: 
?
save/Assign_22Assignlayer11/conv0/biassave/RestoreV2:22*
T0*%
_class
loc:@layer11/conv0/bias*
_output_shapes	
:?
?
save/Assign_23Assignlayer11/conv0/kernelsave/RestoreV2:23*
T0*'
_class
loc:@layer11/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_24Assignlayer11/conv0/wn_gsave/RestoreV2:24*
T0*%
_class
loc:@layer11/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_25Assignlayer11/conv1/biassave/RestoreV2:25*
T0*%
_class
loc:@layer11/conv1/bias*
_output_shapes
: 
?
save/Assign_26Assignlayer11/conv1/kernelsave/RestoreV2:26*'
_output_shapes
:? *
T0*'
_class
loc:@layer11/conv1/kernel
?
save/Assign_27Assignlayer11/conv1/wn_gsave/RestoreV2:27*
T0*%
_class
loc:@layer11/conv1/wn_g*
_output_shapes
: 
?
save/Assign_28Assignlayer12/conv0/biassave/RestoreV2:28*
_output_shapes	
:?*
T0*%
_class
loc:@layer12/conv0/bias
?
save/Assign_29Assignlayer12/conv0/kernelsave/RestoreV2:29*
T0*'
_class
loc:@layer12/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_30Assignlayer12/conv0/wn_gsave/RestoreV2:30*
_output_shapes	
:?*
T0*%
_class
loc:@layer12/conv0/wn_g
?
save/Assign_31Assignlayer12/conv1/biassave/RestoreV2:31*
T0*%
_class
loc:@layer12/conv1/bias*
_output_shapes
: 
?
save/Assign_32Assignlayer12/conv1/kernelsave/RestoreV2:32*
T0*'
_class
loc:@layer12/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_33Assignlayer12/conv1/wn_gsave/RestoreV2:33*
T0*%
_class
loc:@layer12/conv1/wn_g*
_output_shapes
: 
?
save/Assign_34Assignlayer13/conv0/biassave/RestoreV2:34*
T0*%
_class
loc:@layer13/conv0/bias*
_output_shapes	
:?
?
save/Assign_35Assignlayer13/conv0/kernelsave/RestoreV2:35*'
_output_shapes
: ?*
T0*'
_class
loc:@layer13/conv0/kernel
?
save/Assign_36Assignlayer13/conv0/wn_gsave/RestoreV2:36*
T0*%
_class
loc:@layer13/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_37Assignlayer13/conv1/biassave/RestoreV2:37*
T0*%
_class
loc:@layer13/conv1/bias*
_output_shapes
: 
?
save/Assign_38Assignlayer13/conv1/kernelsave/RestoreV2:38*'
_output_shapes
:? *
T0*'
_class
loc:@layer13/conv1/kernel
?
save/Assign_39Assignlayer13/conv1/wn_gsave/RestoreV2:39*
T0*%
_class
loc:@layer13/conv1/wn_g*
_output_shapes
: 
?
save/Assign_40Assignlayer14/conv0/biassave/RestoreV2:40*
T0*%
_class
loc:@layer14/conv0/bias*
_output_shapes	
:?
?
save/Assign_41Assignlayer14/conv0/kernelsave/RestoreV2:41*
T0*'
_class
loc:@layer14/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_42Assignlayer14/conv0/wn_gsave/RestoreV2:42*
T0*%
_class
loc:@layer14/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_43Assignlayer14/conv1/biassave/RestoreV2:43*
T0*%
_class
loc:@layer14/conv1/bias*
_output_shapes
: 
?
save/Assign_44Assignlayer14/conv1/kernelsave/RestoreV2:44*
T0*'
_class
loc:@layer14/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_45Assignlayer14/conv1/wn_gsave/RestoreV2:45*
T0*%
_class
loc:@layer14/conv1/wn_g*
_output_shapes
: 
?
save/Assign_46Assignlayer15/conv0/biassave/RestoreV2:46*
_output_shapes	
:?*
T0*%
_class
loc:@layer15/conv0/bias
?
save/Assign_47Assignlayer15/conv0/kernelsave/RestoreV2:47*
T0*'
_class
loc:@layer15/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_48Assignlayer15/conv0/wn_gsave/RestoreV2:48*
T0*%
_class
loc:@layer15/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_49Assignlayer15/conv1/biassave/RestoreV2:49*
T0*%
_class
loc:@layer15/conv1/bias*
_output_shapes
: 
?
save/Assign_50Assignlayer15/conv1/kernelsave/RestoreV2:50*
T0*'
_class
loc:@layer15/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_51Assignlayer15/conv1/wn_gsave/RestoreV2:51*
T0*%
_class
loc:@layer15/conv1/wn_g*
_output_shapes
: 
?
save/Assign_52Assignlayer2/conv0/biassave/RestoreV2:52*
_output_shapes	
:?*
T0*$
_class
loc:@layer2/conv0/bias
?
save/Assign_53Assignlayer2/conv0/kernelsave/RestoreV2:53*
T0*&
_class
loc:@layer2/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_54Assignlayer2/conv0/wn_gsave/RestoreV2:54*
T0*$
_class
loc:@layer2/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_55Assignlayer2/conv1/biassave/RestoreV2:55*
T0*$
_class
loc:@layer2/conv1/bias*
_output_shapes
: 
?
save/Assign_56Assignlayer2/conv1/kernelsave/RestoreV2:56*
T0*&
_class
loc:@layer2/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_57Assignlayer2/conv1/wn_gsave/RestoreV2:57*
T0*$
_class
loc:@layer2/conv1/wn_g*
_output_shapes
: 
?
save/Assign_58Assignlayer3/conv0/biassave/RestoreV2:58*
T0*$
_class
loc:@layer3/conv0/bias*
_output_shapes	
:?
?
save/Assign_59Assignlayer3/conv0/kernelsave/RestoreV2:59*
T0*&
_class
loc:@layer3/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_60Assignlayer3/conv0/wn_gsave/RestoreV2:60*
T0*$
_class
loc:@layer3/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_61Assignlayer3/conv1/biassave/RestoreV2:61*
_output_shapes
: *
T0*$
_class
loc:@layer3/conv1/bias
?
save/Assign_62Assignlayer3/conv1/kernelsave/RestoreV2:62*
T0*&
_class
loc:@layer3/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_63Assignlayer3/conv1/wn_gsave/RestoreV2:63*
T0*$
_class
loc:@layer3/conv1/wn_g*
_output_shapes
: 
?
save/Assign_64Assignlayer4/conv0/biassave/RestoreV2:64*
T0*$
_class
loc:@layer4/conv0/bias*
_output_shapes	
:?
?
save/Assign_65Assignlayer4/conv0/kernelsave/RestoreV2:65*
T0*&
_class
loc:@layer4/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_66Assignlayer4/conv0/wn_gsave/RestoreV2:66*
_output_shapes	
:?*
T0*$
_class
loc:@layer4/conv0/wn_g
?
save/Assign_67Assignlayer4/conv1/biassave/RestoreV2:67*
T0*$
_class
loc:@layer4/conv1/bias*
_output_shapes
: 
?
save/Assign_68Assignlayer4/conv1/kernelsave/RestoreV2:68*'
_output_shapes
:? *
T0*&
_class
loc:@layer4/conv1/kernel
?
save/Assign_69Assignlayer4/conv1/wn_gsave/RestoreV2:69*
_output_shapes
: *
T0*$
_class
loc:@layer4/conv1/wn_g
?
save/Assign_70Assignlayer5/conv0/biassave/RestoreV2:70*
T0*$
_class
loc:@layer5/conv0/bias*
_output_shapes	
:?
?
save/Assign_71Assignlayer5/conv0/kernelsave/RestoreV2:71*
T0*&
_class
loc:@layer5/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_72Assignlayer5/conv0/wn_gsave/RestoreV2:72*
T0*$
_class
loc:@layer5/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_73Assignlayer5/conv1/biassave/RestoreV2:73*
T0*$
_class
loc:@layer5/conv1/bias*
_output_shapes
: 
?
save/Assign_74Assignlayer5/conv1/kernelsave/RestoreV2:74*
T0*&
_class
loc:@layer5/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_75Assignlayer5/conv1/wn_gsave/RestoreV2:75*
T0*$
_class
loc:@layer5/conv1/wn_g*
_output_shapes
: 
?
save/Assign_76Assignlayer6/conv0/biassave/RestoreV2:76*
T0*$
_class
loc:@layer6/conv0/bias*
_output_shapes	
:?
?
save/Assign_77Assignlayer6/conv0/kernelsave/RestoreV2:77*
T0*&
_class
loc:@layer6/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_78Assignlayer6/conv0/wn_gsave/RestoreV2:78*
T0*$
_class
loc:@layer6/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_79Assignlayer6/conv1/biassave/RestoreV2:79*
T0*$
_class
loc:@layer6/conv1/bias*
_output_shapes
: 
?
save/Assign_80Assignlayer6/conv1/kernelsave/RestoreV2:80*
T0*&
_class
loc:@layer6/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_81Assignlayer6/conv1/wn_gsave/RestoreV2:81*
T0*$
_class
loc:@layer6/conv1/wn_g*
_output_shapes
: 
?
save/Assign_82Assignlayer7/conv0/biassave/RestoreV2:82*
_output_shapes	
:?*
T0*$
_class
loc:@layer7/conv0/bias
?
save/Assign_83Assignlayer7/conv0/kernelsave/RestoreV2:83*'
_output_shapes
: ?*
T0*&
_class
loc:@layer7/conv0/kernel
?
save/Assign_84Assignlayer7/conv0/wn_gsave/RestoreV2:84*
T0*$
_class
loc:@layer7/conv0/wn_g*
_output_shapes	
:?
?
save/Assign_85Assignlayer7/conv1/biassave/RestoreV2:85*
T0*$
_class
loc:@layer7/conv1/bias*
_output_shapes
: 
?
save/Assign_86Assignlayer7/conv1/kernelsave/RestoreV2:86*
T0*&
_class
loc:@layer7/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_87Assignlayer7/conv1/wn_gsave/RestoreV2:87*
_output_shapes
: *
T0*$
_class
loc:@layer7/conv1/wn_g
?
save/Assign_88Assignlayer8/conv0/biassave/RestoreV2:88*
T0*$
_class
loc:@layer8/conv0/bias*
_output_shapes	
:?
?
save/Assign_89Assignlayer8/conv0/kernelsave/RestoreV2:89*
T0*&
_class
loc:@layer8/conv0/kernel*'
_output_shapes
: ?
?
save/Assign_90Assignlayer8/conv0/wn_gsave/RestoreV2:90*
_output_shapes	
:?*
T0*$
_class
loc:@layer8/conv0/wn_g
?
save/Assign_91Assignlayer8/conv1/biassave/RestoreV2:91*
_output_shapes
: *
T0*$
_class
loc:@layer8/conv1/bias
?
save/Assign_92Assignlayer8/conv1/kernelsave/RestoreV2:92*
T0*&
_class
loc:@layer8/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_93Assignlayer8/conv1/wn_gsave/RestoreV2:93*
T0*$
_class
loc:@layer8/conv1/wn_g*
_output_shapes
: 
?
save/Assign_94Assignlayer9/conv0/biassave/RestoreV2:94*
T0*$
_class
loc:@layer9/conv0/bias*
_output_shapes	
:?
?
save/Assign_95Assignlayer9/conv0/kernelsave/RestoreV2:95*'
_output_shapes
: ?*
T0*&
_class
loc:@layer9/conv0/kernel
?
save/Assign_96Assignlayer9/conv0/wn_gsave/RestoreV2:96*
_output_shapes	
:?*
T0*$
_class
loc:@layer9/conv0/wn_g
?
save/Assign_97Assignlayer9/conv1/biassave/RestoreV2:97*
_output_shapes
: *
T0*$
_class
loc:@layer9/conv1/bias
?
save/Assign_98Assignlayer9/conv1/kernelsave/RestoreV2:98*
T0*&
_class
loc:@layer9/conv1/kernel*'
_output_shapes
:? 
?
save/Assign_99Assignlayer9/conv1/wn_gsave/RestoreV2:99*
T0*$
_class
loc:@layer9/conv1/wn_g*
_output_shapes
: 
?
save/Assign_100Assignoutput/conv2d_weight_norm/biassave/RestoreV2:100*
T0*1
_class'
%#loc:@output/conv2d_weight_norm/bias*
_output_shapes
:
?
save/Assign_101Assign output/conv2d_weight_norm/kernelsave/RestoreV2:101*
T0*3
_class)
'%loc:@output/conv2d_weight_norm/kernel*&
_output_shapes
: 
?
save/Assign_102Assignoutput/conv2d_weight_norm/wn_gsave/RestoreV2:102*
T0*1
_class'
%#loc:@output/conv2d_weight_norm/wn_g*
_output_shapes
:
?
save/Assign_103Assignskip/conv2d_weight_norm/biassave/RestoreV2:103*
_output_shapes
:*
T0*/
_class%
#!loc:@skip/conv2d_weight_norm/bias
?
save/Assign_104Assignskip/conv2d_weight_norm/kernelsave/RestoreV2:104*&
_output_shapes
:*
T0*1
_class'
%#loc:@skip/conv2d_weight_norm/kernel
?
save/Assign_105Assignskip/conv2d_weight_norm/wn_gsave/RestoreV2:105*
T0*/
_class%
#!loc:@skip/conv2d_weight_norm/wn_g*
_output_shapes
:
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"?h
trainable_variables?h?h
?
skip/conv2d_weight_norm/wn_g:0#skip/conv2d_weight_norm/wn_g/Assign#skip/conv2d_weight_norm/wn_g/read:02/skip/conv2d_weight_norm/wn_g/Initializer/ones:08
?
 skip/conv2d_weight_norm/kernel:0%skip/conv2d_weight_norm/kernel/Assign%skip/conv2d_weight_norm/kernel/read:02;skip/conv2d_weight_norm/kernel/Initializer/random_uniform:08
?
skip/conv2d_weight_norm/bias:0#skip/conv2d_weight_norm/bias/Assign#skip/conv2d_weight_norm/bias/read:020skip/conv2d_weight_norm/bias/Initializer/zeros:08
?
input/conv2d_weight_norm/wn_g:0$input/conv2d_weight_norm/wn_g/Assign$input/conv2d_weight_norm/wn_g/read:020input/conv2d_weight_norm/wn_g/Initializer/ones:08
?
!input/conv2d_weight_norm/kernel:0&input/conv2d_weight_norm/kernel/Assign&input/conv2d_weight_norm/kernel/read:02<input/conv2d_weight_norm/kernel/Initializer/random_uniform:08
?
input/conv2d_weight_norm/bias:0$input/conv2d_weight_norm/bias/Assign$input/conv2d_weight_norm/bias/read:021input/conv2d_weight_norm/bias/Initializer/zeros:08
q
layer0/conv0/wn_g:0layer0/conv0/wn_g/Assignlayer0/conv0/wn_g/read:02$layer0/conv0/wn_g/Initializer/ones:08
?
layer0/conv0/kernel:0layer0/conv0/kernel/Assignlayer0/conv0/kernel/read:020layer0/conv0/kernel/Initializer/random_uniform:08
r
layer0/conv0/bias:0layer0/conv0/bias/Assignlayer0/conv0/bias/read:02%layer0/conv0/bias/Initializer/zeros:08
q
layer0/conv1/wn_g:0layer0/conv1/wn_g/Assignlayer0/conv1/wn_g/read:02$layer0/conv1/wn_g/Initializer/ones:08
?
layer0/conv1/kernel:0layer0/conv1/kernel/Assignlayer0/conv1/kernel/read:020layer0/conv1/kernel/Initializer/random_uniform:08
r
layer0/conv1/bias:0layer0/conv1/bias/Assignlayer0/conv1/bias/read:02%layer0/conv1/bias/Initializer/zeros:08
q
layer1/conv0/wn_g:0layer1/conv0/wn_g/Assignlayer1/conv0/wn_g/read:02$layer1/conv0/wn_g/Initializer/ones:08
?
layer1/conv0/kernel:0layer1/conv0/kernel/Assignlayer1/conv0/kernel/read:020layer1/conv0/kernel/Initializer/random_uniform:08
r
layer1/conv0/bias:0layer1/conv0/bias/Assignlayer1/conv0/bias/read:02%layer1/conv0/bias/Initializer/zeros:08
q
layer1/conv1/wn_g:0layer1/conv1/wn_g/Assignlayer1/conv1/wn_g/read:02$layer1/conv1/wn_g/Initializer/ones:08
?
layer1/conv1/kernel:0layer1/conv1/kernel/Assignlayer1/conv1/kernel/read:020layer1/conv1/kernel/Initializer/random_uniform:08
r
layer1/conv1/bias:0layer1/conv1/bias/Assignlayer1/conv1/bias/read:02%layer1/conv1/bias/Initializer/zeros:08
q
layer2/conv0/wn_g:0layer2/conv0/wn_g/Assignlayer2/conv0/wn_g/read:02$layer2/conv0/wn_g/Initializer/ones:08
?
layer2/conv0/kernel:0layer2/conv0/kernel/Assignlayer2/conv0/kernel/read:020layer2/conv0/kernel/Initializer/random_uniform:08
r
layer2/conv0/bias:0layer2/conv0/bias/Assignlayer2/conv0/bias/read:02%layer2/conv0/bias/Initializer/zeros:08
q
layer2/conv1/wn_g:0layer2/conv1/wn_g/Assignlayer2/conv1/wn_g/read:02$layer2/conv1/wn_g/Initializer/ones:08
?
layer2/conv1/kernel:0layer2/conv1/kernel/Assignlayer2/conv1/kernel/read:020layer2/conv1/kernel/Initializer/random_uniform:08
r
layer2/conv1/bias:0layer2/conv1/bias/Assignlayer2/conv1/bias/read:02%layer2/conv1/bias/Initializer/zeros:08
q
layer3/conv0/wn_g:0layer3/conv0/wn_g/Assignlayer3/conv0/wn_g/read:02$layer3/conv0/wn_g/Initializer/ones:08
?
layer3/conv0/kernel:0layer3/conv0/kernel/Assignlayer3/conv0/kernel/read:020layer3/conv0/kernel/Initializer/random_uniform:08
r
layer3/conv0/bias:0layer3/conv0/bias/Assignlayer3/conv0/bias/read:02%layer3/conv0/bias/Initializer/zeros:08
q
layer3/conv1/wn_g:0layer3/conv1/wn_g/Assignlayer3/conv1/wn_g/read:02$layer3/conv1/wn_g/Initializer/ones:08
?
layer3/conv1/kernel:0layer3/conv1/kernel/Assignlayer3/conv1/kernel/read:020layer3/conv1/kernel/Initializer/random_uniform:08
r
layer3/conv1/bias:0layer3/conv1/bias/Assignlayer3/conv1/bias/read:02%layer3/conv1/bias/Initializer/zeros:08
q
layer4/conv0/wn_g:0layer4/conv0/wn_g/Assignlayer4/conv0/wn_g/read:02$layer4/conv0/wn_g/Initializer/ones:08
?
layer4/conv0/kernel:0layer4/conv0/kernel/Assignlayer4/conv0/kernel/read:020layer4/conv0/kernel/Initializer/random_uniform:08
r
layer4/conv0/bias:0layer4/conv0/bias/Assignlayer4/conv0/bias/read:02%layer4/conv0/bias/Initializer/zeros:08
q
layer4/conv1/wn_g:0layer4/conv1/wn_g/Assignlayer4/conv1/wn_g/read:02$layer4/conv1/wn_g/Initializer/ones:08
?
layer4/conv1/kernel:0layer4/conv1/kernel/Assignlayer4/conv1/kernel/read:020layer4/conv1/kernel/Initializer/random_uniform:08
r
layer4/conv1/bias:0layer4/conv1/bias/Assignlayer4/conv1/bias/read:02%layer4/conv1/bias/Initializer/zeros:08
q
layer5/conv0/wn_g:0layer5/conv0/wn_g/Assignlayer5/conv0/wn_g/read:02$layer5/conv0/wn_g/Initializer/ones:08
?
layer5/conv0/kernel:0layer5/conv0/kernel/Assignlayer5/conv0/kernel/read:020layer5/conv0/kernel/Initializer/random_uniform:08
r
layer5/conv0/bias:0layer5/conv0/bias/Assignlayer5/conv0/bias/read:02%layer5/conv0/bias/Initializer/zeros:08
q
layer5/conv1/wn_g:0layer5/conv1/wn_g/Assignlayer5/conv1/wn_g/read:02$layer5/conv1/wn_g/Initializer/ones:08
?
layer5/conv1/kernel:0layer5/conv1/kernel/Assignlayer5/conv1/kernel/read:020layer5/conv1/kernel/Initializer/random_uniform:08
r
layer5/conv1/bias:0layer5/conv1/bias/Assignlayer5/conv1/bias/read:02%layer5/conv1/bias/Initializer/zeros:08
q
layer6/conv0/wn_g:0layer6/conv0/wn_g/Assignlayer6/conv0/wn_g/read:02$layer6/conv0/wn_g/Initializer/ones:08
?
layer6/conv0/kernel:0layer6/conv0/kernel/Assignlayer6/conv0/kernel/read:020layer6/conv0/kernel/Initializer/random_uniform:08
r
layer6/conv0/bias:0layer6/conv0/bias/Assignlayer6/conv0/bias/read:02%layer6/conv0/bias/Initializer/zeros:08
q
layer6/conv1/wn_g:0layer6/conv1/wn_g/Assignlayer6/conv1/wn_g/read:02$layer6/conv1/wn_g/Initializer/ones:08
?
layer6/conv1/kernel:0layer6/conv1/kernel/Assignlayer6/conv1/kernel/read:020layer6/conv1/kernel/Initializer/random_uniform:08
r
layer6/conv1/bias:0layer6/conv1/bias/Assignlayer6/conv1/bias/read:02%layer6/conv1/bias/Initializer/zeros:08
q
layer7/conv0/wn_g:0layer7/conv0/wn_g/Assignlayer7/conv0/wn_g/read:02$layer7/conv0/wn_g/Initializer/ones:08
?
layer7/conv0/kernel:0layer7/conv0/kernel/Assignlayer7/conv0/kernel/read:020layer7/conv0/kernel/Initializer/random_uniform:08
r
layer7/conv0/bias:0layer7/conv0/bias/Assignlayer7/conv0/bias/read:02%layer7/conv0/bias/Initializer/zeros:08
q
layer7/conv1/wn_g:0layer7/conv1/wn_g/Assignlayer7/conv1/wn_g/read:02$layer7/conv1/wn_g/Initializer/ones:08
?
layer7/conv1/kernel:0layer7/conv1/kernel/Assignlayer7/conv1/kernel/read:020layer7/conv1/kernel/Initializer/random_uniform:08
r
layer7/conv1/bias:0layer7/conv1/bias/Assignlayer7/conv1/bias/read:02%layer7/conv1/bias/Initializer/zeros:08
q
layer8/conv0/wn_g:0layer8/conv0/wn_g/Assignlayer8/conv0/wn_g/read:02$layer8/conv0/wn_g/Initializer/ones:08
?
layer8/conv0/kernel:0layer8/conv0/kernel/Assignlayer8/conv0/kernel/read:020layer8/conv0/kernel/Initializer/random_uniform:08
r
layer8/conv0/bias:0layer8/conv0/bias/Assignlayer8/conv0/bias/read:02%layer8/conv0/bias/Initializer/zeros:08
q
layer8/conv1/wn_g:0layer8/conv1/wn_g/Assignlayer8/conv1/wn_g/read:02$layer8/conv1/wn_g/Initializer/ones:08
?
layer8/conv1/kernel:0layer8/conv1/kernel/Assignlayer8/conv1/kernel/read:020layer8/conv1/kernel/Initializer/random_uniform:08
r
layer8/conv1/bias:0layer8/conv1/bias/Assignlayer8/conv1/bias/read:02%layer8/conv1/bias/Initializer/zeros:08
q
layer9/conv0/wn_g:0layer9/conv0/wn_g/Assignlayer9/conv0/wn_g/read:02$layer9/conv0/wn_g/Initializer/ones:08
?
layer9/conv0/kernel:0layer9/conv0/kernel/Assignlayer9/conv0/kernel/read:020layer9/conv0/kernel/Initializer/random_uniform:08
r
layer9/conv0/bias:0layer9/conv0/bias/Assignlayer9/conv0/bias/read:02%layer9/conv0/bias/Initializer/zeros:08
q
layer9/conv1/wn_g:0layer9/conv1/wn_g/Assignlayer9/conv1/wn_g/read:02$layer9/conv1/wn_g/Initializer/ones:08
?
layer9/conv1/kernel:0layer9/conv1/kernel/Assignlayer9/conv1/kernel/read:020layer9/conv1/kernel/Initializer/random_uniform:08
r
layer9/conv1/bias:0layer9/conv1/bias/Assignlayer9/conv1/bias/read:02%layer9/conv1/bias/Initializer/zeros:08
u
layer10/conv0/wn_g:0layer10/conv0/wn_g/Assignlayer10/conv0/wn_g/read:02%layer10/conv0/wn_g/Initializer/ones:08
?
layer10/conv0/kernel:0layer10/conv0/kernel/Assignlayer10/conv0/kernel/read:021layer10/conv0/kernel/Initializer/random_uniform:08
v
layer10/conv0/bias:0layer10/conv0/bias/Assignlayer10/conv0/bias/read:02&layer10/conv0/bias/Initializer/zeros:08
u
layer10/conv1/wn_g:0layer10/conv1/wn_g/Assignlayer10/conv1/wn_g/read:02%layer10/conv1/wn_g/Initializer/ones:08
?
layer10/conv1/kernel:0layer10/conv1/kernel/Assignlayer10/conv1/kernel/read:021layer10/conv1/kernel/Initializer/random_uniform:08
v
layer10/conv1/bias:0layer10/conv1/bias/Assignlayer10/conv1/bias/read:02&layer10/conv1/bias/Initializer/zeros:08
u
layer11/conv0/wn_g:0layer11/conv0/wn_g/Assignlayer11/conv0/wn_g/read:02%layer11/conv0/wn_g/Initializer/ones:08
?
layer11/conv0/kernel:0layer11/conv0/kernel/Assignlayer11/conv0/kernel/read:021layer11/conv0/kernel/Initializer/random_uniform:08
v
layer11/conv0/bias:0layer11/conv0/bias/Assignlayer11/conv0/bias/read:02&layer11/conv0/bias/Initializer/zeros:08
u
layer11/conv1/wn_g:0layer11/conv1/wn_g/Assignlayer11/conv1/wn_g/read:02%layer11/conv1/wn_g/Initializer/ones:08
?
layer11/conv1/kernel:0layer11/conv1/kernel/Assignlayer11/conv1/kernel/read:021layer11/conv1/kernel/Initializer/random_uniform:08
v
layer11/conv1/bias:0layer11/conv1/bias/Assignlayer11/conv1/bias/read:02&layer11/conv1/bias/Initializer/zeros:08
u
layer12/conv0/wn_g:0layer12/conv0/wn_g/Assignlayer12/conv0/wn_g/read:02%layer12/conv0/wn_g/Initializer/ones:08
?
layer12/conv0/kernel:0layer12/conv0/kernel/Assignlayer12/conv0/kernel/read:021layer12/conv0/kernel/Initializer/random_uniform:08
v
layer12/conv0/bias:0layer12/conv0/bias/Assignlayer12/conv0/bias/read:02&layer12/conv0/bias/Initializer/zeros:08
u
layer12/conv1/wn_g:0layer12/conv1/wn_g/Assignlayer12/conv1/wn_g/read:02%layer12/conv1/wn_g/Initializer/ones:08
?
layer12/conv1/kernel:0layer12/conv1/kernel/Assignlayer12/conv1/kernel/read:021layer12/conv1/kernel/Initializer/random_uniform:08
v
layer12/conv1/bias:0layer12/conv1/bias/Assignlayer12/conv1/bias/read:02&layer12/conv1/bias/Initializer/zeros:08
u
layer13/conv0/wn_g:0layer13/conv0/wn_g/Assignlayer13/conv0/wn_g/read:02%layer13/conv0/wn_g/Initializer/ones:08
?
layer13/conv0/kernel:0layer13/conv0/kernel/Assignlayer13/conv0/kernel/read:021layer13/conv0/kernel/Initializer/random_uniform:08
v
layer13/conv0/bias:0layer13/conv0/bias/Assignlayer13/conv0/bias/read:02&layer13/conv0/bias/Initializer/zeros:08
u
layer13/conv1/wn_g:0layer13/conv1/wn_g/Assignlayer13/conv1/wn_g/read:02%layer13/conv1/wn_g/Initializer/ones:08
?
layer13/conv1/kernel:0layer13/conv1/kernel/Assignlayer13/conv1/kernel/read:021layer13/conv1/kernel/Initializer/random_uniform:08
v
layer13/conv1/bias:0layer13/conv1/bias/Assignlayer13/conv1/bias/read:02&layer13/conv1/bias/Initializer/zeros:08
u
layer14/conv0/wn_g:0layer14/conv0/wn_g/Assignlayer14/conv0/wn_g/read:02%layer14/conv0/wn_g/Initializer/ones:08
?
layer14/conv0/kernel:0layer14/conv0/kernel/Assignlayer14/conv0/kernel/read:021layer14/conv0/kernel/Initializer/random_uniform:08
v
layer14/conv0/bias:0layer14/conv0/bias/Assignlayer14/conv0/bias/read:02&layer14/conv0/bias/Initializer/zeros:08
u
layer14/conv1/wn_g:0layer14/conv1/wn_g/Assignlayer14/conv1/wn_g/read:02%layer14/conv1/wn_g/Initializer/ones:08
?
layer14/conv1/kernel:0layer14/conv1/kernel/Assignlayer14/conv1/kernel/read:021layer14/conv1/kernel/Initializer/random_uniform:08
v
layer14/conv1/bias:0layer14/conv1/bias/Assignlayer14/conv1/bias/read:02&layer14/conv1/bias/Initializer/zeros:08
u
layer15/conv0/wn_g:0layer15/conv0/wn_g/Assignlayer15/conv0/wn_g/read:02%layer15/conv0/wn_g/Initializer/ones:08
?
layer15/conv0/kernel:0layer15/conv0/kernel/Assignlayer15/conv0/kernel/read:021layer15/conv0/kernel/Initializer/random_uniform:08
v
layer15/conv0/bias:0layer15/conv0/bias/Assignlayer15/conv0/bias/read:02&layer15/conv0/bias/Initializer/zeros:08
u
layer15/conv1/wn_g:0layer15/conv1/wn_g/Assignlayer15/conv1/wn_g/read:02%layer15/conv1/wn_g/Initializer/ones:08
?
layer15/conv1/kernel:0layer15/conv1/kernel/Assignlayer15/conv1/kernel/read:021layer15/conv1/kernel/Initializer/random_uniform:08
v
layer15/conv1/bias:0layer15/conv1/bias/Assignlayer15/conv1/bias/read:02&layer15/conv1/bias/Initializer/zeros:08
?
 output/conv2d_weight_norm/wn_g:0%output/conv2d_weight_norm/wn_g/Assign%output/conv2d_weight_norm/wn_g/read:021output/conv2d_weight_norm/wn_g/Initializer/ones:08
?
"output/conv2d_weight_norm/kernel:0'output/conv2d_weight_norm/kernel/Assign'output/conv2d_weight_norm/kernel/read:02=output/conv2d_weight_norm/kernel/Initializer/random_uniform:08
?
 output/conv2d_weight_norm/bias:0%output/conv2d_weight_norm/bias/Assign%output/conv2d_weight_norm/bias/read:022output/conv2d_weight_norm/bias/Initializer/zeros:08"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"%
saved_model_main_op


group_deps"?i
	variables?i?i
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
?
skip/conv2d_weight_norm/wn_g:0#skip/conv2d_weight_norm/wn_g/Assign#skip/conv2d_weight_norm/wn_g/read:02/skip/conv2d_weight_norm/wn_g/Initializer/ones:08
?
 skip/conv2d_weight_norm/kernel:0%skip/conv2d_weight_norm/kernel/Assign%skip/conv2d_weight_norm/kernel/read:02;skip/conv2d_weight_norm/kernel/Initializer/random_uniform:08
?
skip/conv2d_weight_norm/bias:0#skip/conv2d_weight_norm/bias/Assign#skip/conv2d_weight_norm/bias/read:020skip/conv2d_weight_norm/bias/Initializer/zeros:08
?
input/conv2d_weight_norm/wn_g:0$input/conv2d_weight_norm/wn_g/Assign$input/conv2d_weight_norm/wn_g/read:020input/conv2d_weight_norm/wn_g/Initializer/ones:08
?
!input/conv2d_weight_norm/kernel:0&input/conv2d_weight_norm/kernel/Assign&input/conv2d_weight_norm/kernel/read:02<input/conv2d_weight_norm/kernel/Initializer/random_uniform:08
?
input/conv2d_weight_norm/bias:0$input/conv2d_weight_norm/bias/Assign$input/conv2d_weight_norm/bias/read:021input/conv2d_weight_norm/bias/Initializer/zeros:08
q
layer0/conv0/wn_g:0layer0/conv0/wn_g/Assignlayer0/conv0/wn_g/read:02$layer0/conv0/wn_g/Initializer/ones:08
?
layer0/conv0/kernel:0layer0/conv0/kernel/Assignlayer0/conv0/kernel/read:020layer0/conv0/kernel/Initializer/random_uniform:08
r
layer0/conv0/bias:0layer0/conv0/bias/Assignlayer0/conv0/bias/read:02%layer0/conv0/bias/Initializer/zeros:08
q
layer0/conv1/wn_g:0layer0/conv1/wn_g/Assignlayer0/conv1/wn_g/read:02$layer0/conv1/wn_g/Initializer/ones:08
?
layer0/conv1/kernel:0layer0/conv1/kernel/Assignlayer0/conv1/kernel/read:020layer0/conv1/kernel/Initializer/random_uniform:08
r
layer0/conv1/bias:0layer0/conv1/bias/Assignlayer0/conv1/bias/read:02%layer0/conv1/bias/Initializer/zeros:08
q
layer1/conv0/wn_g:0layer1/conv0/wn_g/Assignlayer1/conv0/wn_g/read:02$layer1/conv0/wn_g/Initializer/ones:08
?
layer1/conv0/kernel:0layer1/conv0/kernel/Assignlayer1/conv0/kernel/read:020layer1/conv0/kernel/Initializer/random_uniform:08
r
layer1/conv0/bias:0layer1/conv0/bias/Assignlayer1/conv0/bias/read:02%layer1/conv0/bias/Initializer/zeros:08
q
layer1/conv1/wn_g:0layer1/conv1/wn_g/Assignlayer1/conv1/wn_g/read:02$layer1/conv1/wn_g/Initializer/ones:08
?
layer1/conv1/kernel:0layer1/conv1/kernel/Assignlayer1/conv1/kernel/read:020layer1/conv1/kernel/Initializer/random_uniform:08
r
layer1/conv1/bias:0layer1/conv1/bias/Assignlayer1/conv1/bias/read:02%layer1/conv1/bias/Initializer/zeros:08
q
layer2/conv0/wn_g:0layer2/conv0/wn_g/Assignlayer2/conv0/wn_g/read:02$layer2/conv0/wn_g/Initializer/ones:08
?
layer2/conv0/kernel:0layer2/conv0/kernel/Assignlayer2/conv0/kernel/read:020layer2/conv0/kernel/Initializer/random_uniform:08
r
layer2/conv0/bias:0layer2/conv0/bias/Assignlayer2/conv0/bias/read:02%layer2/conv0/bias/Initializer/zeros:08
q
layer2/conv1/wn_g:0layer2/conv1/wn_g/Assignlayer2/conv1/wn_g/read:02$layer2/conv1/wn_g/Initializer/ones:08
?
layer2/conv1/kernel:0layer2/conv1/kernel/Assignlayer2/conv1/kernel/read:020layer2/conv1/kernel/Initializer/random_uniform:08
r
layer2/conv1/bias:0layer2/conv1/bias/Assignlayer2/conv1/bias/read:02%layer2/conv1/bias/Initializer/zeros:08
q
layer3/conv0/wn_g:0layer3/conv0/wn_g/Assignlayer3/conv0/wn_g/read:02$layer3/conv0/wn_g/Initializer/ones:08
?
layer3/conv0/kernel:0layer3/conv0/kernel/Assignlayer3/conv0/kernel/read:020layer3/conv0/kernel/Initializer/random_uniform:08
r
layer3/conv0/bias:0layer3/conv0/bias/Assignlayer3/conv0/bias/read:02%layer3/conv0/bias/Initializer/zeros:08
q
layer3/conv1/wn_g:0layer3/conv1/wn_g/Assignlayer3/conv1/wn_g/read:02$layer3/conv1/wn_g/Initializer/ones:08
?
layer3/conv1/kernel:0layer3/conv1/kernel/Assignlayer3/conv1/kernel/read:020layer3/conv1/kernel/Initializer/random_uniform:08
r
layer3/conv1/bias:0layer3/conv1/bias/Assignlayer3/conv1/bias/read:02%layer3/conv1/bias/Initializer/zeros:08
q
layer4/conv0/wn_g:0layer4/conv0/wn_g/Assignlayer4/conv0/wn_g/read:02$layer4/conv0/wn_g/Initializer/ones:08
?
layer4/conv0/kernel:0layer4/conv0/kernel/Assignlayer4/conv0/kernel/read:020layer4/conv0/kernel/Initializer/random_uniform:08
r
layer4/conv0/bias:0layer4/conv0/bias/Assignlayer4/conv0/bias/read:02%layer4/conv0/bias/Initializer/zeros:08
q
layer4/conv1/wn_g:0layer4/conv1/wn_g/Assignlayer4/conv1/wn_g/read:02$layer4/conv1/wn_g/Initializer/ones:08
?
layer4/conv1/kernel:0layer4/conv1/kernel/Assignlayer4/conv1/kernel/read:020layer4/conv1/kernel/Initializer/random_uniform:08
r
layer4/conv1/bias:0layer4/conv1/bias/Assignlayer4/conv1/bias/read:02%layer4/conv1/bias/Initializer/zeros:08
q
layer5/conv0/wn_g:0layer5/conv0/wn_g/Assignlayer5/conv0/wn_g/read:02$layer5/conv0/wn_g/Initializer/ones:08
?
layer5/conv0/kernel:0layer5/conv0/kernel/Assignlayer5/conv0/kernel/read:020layer5/conv0/kernel/Initializer/random_uniform:08
r
layer5/conv0/bias:0layer5/conv0/bias/Assignlayer5/conv0/bias/read:02%layer5/conv0/bias/Initializer/zeros:08
q
layer5/conv1/wn_g:0layer5/conv1/wn_g/Assignlayer5/conv1/wn_g/read:02$layer5/conv1/wn_g/Initializer/ones:08
?
layer5/conv1/kernel:0layer5/conv1/kernel/Assignlayer5/conv1/kernel/read:020layer5/conv1/kernel/Initializer/random_uniform:08
r
layer5/conv1/bias:0layer5/conv1/bias/Assignlayer5/conv1/bias/read:02%layer5/conv1/bias/Initializer/zeros:08
q
layer6/conv0/wn_g:0layer6/conv0/wn_g/Assignlayer6/conv0/wn_g/read:02$layer6/conv0/wn_g/Initializer/ones:08
?
layer6/conv0/kernel:0layer6/conv0/kernel/Assignlayer6/conv0/kernel/read:020layer6/conv0/kernel/Initializer/random_uniform:08
r
layer6/conv0/bias:0layer6/conv0/bias/Assignlayer6/conv0/bias/read:02%layer6/conv0/bias/Initializer/zeros:08
q
layer6/conv1/wn_g:0layer6/conv1/wn_g/Assignlayer6/conv1/wn_g/read:02$layer6/conv1/wn_g/Initializer/ones:08
?
layer6/conv1/kernel:0layer6/conv1/kernel/Assignlayer6/conv1/kernel/read:020layer6/conv1/kernel/Initializer/random_uniform:08
r
layer6/conv1/bias:0layer6/conv1/bias/Assignlayer6/conv1/bias/read:02%layer6/conv1/bias/Initializer/zeros:08
q
layer7/conv0/wn_g:0layer7/conv0/wn_g/Assignlayer7/conv0/wn_g/read:02$layer7/conv0/wn_g/Initializer/ones:08
?
layer7/conv0/kernel:0layer7/conv0/kernel/Assignlayer7/conv0/kernel/read:020layer7/conv0/kernel/Initializer/random_uniform:08
r
layer7/conv0/bias:0layer7/conv0/bias/Assignlayer7/conv0/bias/read:02%layer7/conv0/bias/Initializer/zeros:08
q
layer7/conv1/wn_g:0layer7/conv1/wn_g/Assignlayer7/conv1/wn_g/read:02$layer7/conv1/wn_g/Initializer/ones:08
?
layer7/conv1/kernel:0layer7/conv1/kernel/Assignlayer7/conv1/kernel/read:020layer7/conv1/kernel/Initializer/random_uniform:08
r
layer7/conv1/bias:0layer7/conv1/bias/Assignlayer7/conv1/bias/read:02%layer7/conv1/bias/Initializer/zeros:08
q
layer8/conv0/wn_g:0layer8/conv0/wn_g/Assignlayer8/conv0/wn_g/read:02$layer8/conv0/wn_g/Initializer/ones:08
?
layer8/conv0/kernel:0layer8/conv0/kernel/Assignlayer8/conv0/kernel/read:020layer8/conv0/kernel/Initializer/random_uniform:08
r
layer8/conv0/bias:0layer8/conv0/bias/Assignlayer8/conv0/bias/read:02%layer8/conv0/bias/Initializer/zeros:08
q
layer8/conv1/wn_g:0layer8/conv1/wn_g/Assignlayer8/conv1/wn_g/read:02$layer8/conv1/wn_g/Initializer/ones:08
?
layer8/conv1/kernel:0layer8/conv1/kernel/Assignlayer8/conv1/kernel/read:020layer8/conv1/kernel/Initializer/random_uniform:08
r
layer8/conv1/bias:0layer8/conv1/bias/Assignlayer8/conv1/bias/read:02%layer8/conv1/bias/Initializer/zeros:08
q
layer9/conv0/wn_g:0layer9/conv0/wn_g/Assignlayer9/conv0/wn_g/read:02$layer9/conv0/wn_g/Initializer/ones:08
?
layer9/conv0/kernel:0layer9/conv0/kernel/Assignlayer9/conv0/kernel/read:020layer9/conv0/kernel/Initializer/random_uniform:08
r
layer9/conv0/bias:0layer9/conv0/bias/Assignlayer9/conv0/bias/read:02%layer9/conv0/bias/Initializer/zeros:08
q
layer9/conv1/wn_g:0layer9/conv1/wn_g/Assignlayer9/conv1/wn_g/read:02$layer9/conv1/wn_g/Initializer/ones:08
?
layer9/conv1/kernel:0layer9/conv1/kernel/Assignlayer9/conv1/kernel/read:020layer9/conv1/kernel/Initializer/random_uniform:08
r
layer9/conv1/bias:0layer9/conv1/bias/Assignlayer9/conv1/bias/read:02%layer9/conv1/bias/Initializer/zeros:08
u
layer10/conv0/wn_g:0layer10/conv0/wn_g/Assignlayer10/conv0/wn_g/read:02%layer10/conv0/wn_g/Initializer/ones:08
?
layer10/conv0/kernel:0layer10/conv0/kernel/Assignlayer10/conv0/kernel/read:021layer10/conv0/kernel/Initializer/random_uniform:08
v
layer10/conv0/bias:0layer10/conv0/bias/Assignlayer10/conv0/bias/read:02&layer10/conv0/bias/Initializer/zeros:08
u
layer10/conv1/wn_g:0layer10/conv1/wn_g/Assignlayer10/conv1/wn_g/read:02%layer10/conv1/wn_g/Initializer/ones:08
?
layer10/conv1/kernel:0layer10/conv1/kernel/Assignlayer10/conv1/kernel/read:021layer10/conv1/kernel/Initializer/random_uniform:08
v
layer10/conv1/bias:0layer10/conv1/bias/Assignlayer10/conv1/bias/read:02&layer10/conv1/bias/Initializer/zeros:08
u
layer11/conv0/wn_g:0layer11/conv0/wn_g/Assignlayer11/conv0/wn_g/read:02%layer11/conv0/wn_g/Initializer/ones:08
?
layer11/conv0/kernel:0layer11/conv0/kernel/Assignlayer11/conv0/kernel/read:021layer11/conv0/kernel/Initializer/random_uniform:08
v
layer11/conv0/bias:0layer11/conv0/bias/Assignlayer11/conv0/bias/read:02&layer11/conv0/bias/Initializer/zeros:08
u
layer11/conv1/wn_g:0layer11/conv1/wn_g/Assignlayer11/conv1/wn_g/read:02%layer11/conv1/wn_g/Initializer/ones:08
?
layer11/conv1/kernel:0layer11/conv1/kernel/Assignlayer11/conv1/kernel/read:021layer11/conv1/kernel/Initializer/random_uniform:08
v
layer11/conv1/bias:0layer11/conv1/bias/Assignlayer11/conv1/bias/read:02&layer11/conv1/bias/Initializer/zeros:08
u
layer12/conv0/wn_g:0layer12/conv0/wn_g/Assignlayer12/conv0/wn_g/read:02%layer12/conv0/wn_g/Initializer/ones:08
?
layer12/conv0/kernel:0layer12/conv0/kernel/Assignlayer12/conv0/kernel/read:021layer12/conv0/kernel/Initializer/random_uniform:08
v
layer12/conv0/bias:0layer12/conv0/bias/Assignlayer12/conv0/bias/read:02&layer12/conv0/bias/Initializer/zeros:08
u
layer12/conv1/wn_g:0layer12/conv1/wn_g/Assignlayer12/conv1/wn_g/read:02%layer12/conv1/wn_g/Initializer/ones:08
?
layer12/conv1/kernel:0layer12/conv1/kernel/Assignlayer12/conv1/kernel/read:021layer12/conv1/kernel/Initializer/random_uniform:08
v
layer12/conv1/bias:0layer12/conv1/bias/Assignlayer12/conv1/bias/read:02&layer12/conv1/bias/Initializer/zeros:08
u
layer13/conv0/wn_g:0layer13/conv0/wn_g/Assignlayer13/conv0/wn_g/read:02%layer13/conv0/wn_g/Initializer/ones:08
?
layer13/conv0/kernel:0layer13/conv0/kernel/Assignlayer13/conv0/kernel/read:021layer13/conv0/kernel/Initializer/random_uniform:08
v
layer13/conv0/bias:0layer13/conv0/bias/Assignlayer13/conv0/bias/read:02&layer13/conv0/bias/Initializer/zeros:08
u
layer13/conv1/wn_g:0layer13/conv1/wn_g/Assignlayer13/conv1/wn_g/read:02%layer13/conv1/wn_g/Initializer/ones:08
?
layer13/conv1/kernel:0layer13/conv1/kernel/Assignlayer13/conv1/kernel/read:021layer13/conv1/kernel/Initializer/random_uniform:08
v
layer13/conv1/bias:0layer13/conv1/bias/Assignlayer13/conv1/bias/read:02&layer13/conv1/bias/Initializer/zeros:08
u
layer14/conv0/wn_g:0layer14/conv0/wn_g/Assignlayer14/conv0/wn_g/read:02%layer14/conv0/wn_g/Initializer/ones:08
?
layer14/conv0/kernel:0layer14/conv0/kernel/Assignlayer14/conv0/kernel/read:021layer14/conv0/kernel/Initializer/random_uniform:08
v
layer14/conv0/bias:0layer14/conv0/bias/Assignlayer14/conv0/bias/read:02&layer14/conv0/bias/Initializer/zeros:08
u
layer14/conv1/wn_g:0layer14/conv1/wn_g/Assignlayer14/conv1/wn_g/read:02%layer14/conv1/wn_g/Initializer/ones:08
?
layer14/conv1/kernel:0layer14/conv1/kernel/Assignlayer14/conv1/kernel/read:021layer14/conv1/kernel/Initializer/random_uniform:08
v
layer14/conv1/bias:0layer14/conv1/bias/Assignlayer14/conv1/bias/read:02&layer14/conv1/bias/Initializer/zeros:08
u
layer15/conv0/wn_g:0layer15/conv0/wn_g/Assignlayer15/conv0/wn_g/read:02%layer15/conv0/wn_g/Initializer/ones:08
?
layer15/conv0/kernel:0layer15/conv0/kernel/Assignlayer15/conv0/kernel/read:021layer15/conv0/kernel/Initializer/random_uniform:08
v
layer15/conv0/bias:0layer15/conv0/bias/Assignlayer15/conv0/bias/read:02&layer15/conv0/bias/Initializer/zeros:08
u
layer15/conv1/wn_g:0layer15/conv1/wn_g/Assignlayer15/conv1/wn_g/read:02%layer15/conv1/wn_g/Initializer/ones:08
?
layer15/conv1/kernel:0layer15/conv1/kernel/Assignlayer15/conv1/kernel/read:021layer15/conv1/kernel/Initializer/random_uniform:08
v
layer15/conv1/bias:0layer15/conv1/bias/Assignlayer15/conv1/bias/read:02&layer15/conv1/bias/Initializer/zeros:08
?
 output/conv2d_weight_norm/wn_g:0%output/conv2d_weight_norm/wn_g/Assign%output/conv2d_weight_norm/wn_g/read:021output/conv2d_weight_norm/wn_g/Initializer/ones:08
?
"output/conv2d_weight_norm/kernel:0'output/conv2d_weight_norm/kernel/Assign'output/conv2d_weight_norm/kernel/read:02=output/conv2d_weight_norm/kernel/Initializer/random_uniform:08
?
 output/conv2d_weight_norm/bias:0%output/conv2d_weight_norm/bias/Assign%output/conv2d_weight_norm/bias/read:022output/conv2d_weight_norm/bias/Initializer/zeros:08*?
serving_default?
I
inputs?
input_tensor:0+???????????????????????????J
output@
clip_by_value:0+???????????????????????????tensorflow/serving/predict*?
tensorflow/serving/predict?
I
inputs?
input_tensor:0+???????????????????????????J
output@
clip_by_value:0+???????????????????????????tensorflow/serving/predict