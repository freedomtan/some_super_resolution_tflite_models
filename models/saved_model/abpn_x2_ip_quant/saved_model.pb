¾)
¬ù
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


DepthToSpace

input"T
output"T"	
Ttype"

block_sizeint(0":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
s
FakeQuantWithMinMaxVars

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
}
!FakeQuantWithMinMaxVarsPerChannel

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
.
Identity

input"T
output"T"	
Ttype
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.021.15.0-rc1-63733-g9b0188dedd58Õè$
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
|
quant_add/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namequant_add/output_max
u
(quant_add/output_max/Read/ReadVariableOpReadVariableOpquant_add/output_max*
_output_shapes
: *
dtype0
|
quant_add/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namequant_add/output_min
u
(quant_add/output_min/Read/ReadVariableOpReadVariableOpquant_add/output_min*
_output_shapes
: *
dtype0

quant_add/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_add/optimizer_step
}
,quant_add/optimizer_step/Read/ReadVariableOpReadVariableOpquant_add/optimizer_step*
_output_shapes
: *
dtype0

"quant_conv2d_6/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_6/post_activation_max

6quant_conv2d_6/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_6/post_activation_max*
_output_shapes
: *
dtype0

"quant_conv2d_6/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_6/post_activation_min

6quant_conv2d_6/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_6/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d_6/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_6/kernel_max

-quant_conv2d_6/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_6/kernel_max*
_output_shapes
:*
dtype0

quant_conv2d_6/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_6/kernel_min

-quant_conv2d_6/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_6/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_6/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_6/optimizer_step

1quant_conv2d_6/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_6/optimizer_step*
_output_shapes
: *
dtype0

"quant_conv2d_5/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_5/post_activation_max

6quant_conv2d_5/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_5/post_activation_max*
_output_shapes
: *
dtype0

"quant_conv2d_5/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_5/post_activation_min

6quant_conv2d_5/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_5/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d_5/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_5/kernel_max

-quant_conv2d_5/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_5/kernel_max*
_output_shapes
:*
dtype0

quant_conv2d_5/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_5/kernel_min

-quant_conv2d_5/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_5/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_5/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_5/optimizer_step

1quant_conv2d_5/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_5/optimizer_step*
_output_shapes
: *
dtype0

"quant_conv2d_4/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_4/post_activation_max

6quant_conv2d_4/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_4/post_activation_max*
_output_shapes
: *
dtype0

"quant_conv2d_4/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_4/post_activation_min

6quant_conv2d_4/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_4/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d_4/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_4/kernel_max

-quant_conv2d_4/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_4/kernel_max*
_output_shapes
:*
dtype0

quant_conv2d_4/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_4/kernel_min

-quant_conv2d_4/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_4/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_4/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_4/optimizer_step

1quant_conv2d_4/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_4/optimizer_step*
_output_shapes
: *
dtype0

"quant_conv2d_3/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_3/post_activation_max

6quant_conv2d_3/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_3/post_activation_max*
_output_shapes
: *
dtype0

"quant_conv2d_3/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_3/post_activation_min

6quant_conv2d_3/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_3/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d_3/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_3/kernel_max

-quant_conv2d_3/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_3/kernel_max*
_output_shapes
:*
dtype0

quant_conv2d_3/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_3/kernel_min

-quant_conv2d_3/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_3/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_3/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_3/optimizer_step

1quant_conv2d_3/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_3/optimizer_step*
_output_shapes
: *
dtype0

"quant_conv2d_2/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_2/post_activation_max

6quant_conv2d_2/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_2/post_activation_max*
_output_shapes
: *
dtype0

"quant_conv2d_2/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_2/post_activation_min

6quant_conv2d_2/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_2/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d_2/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_2/kernel_max

-quant_conv2d_2/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_2/kernel_max*
_output_shapes
:*
dtype0

quant_conv2d_2/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_2/kernel_min

-quant_conv2d_2/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_2/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_2/optimizer_step

1quant_conv2d_2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_2/optimizer_step*
_output_shapes
: *
dtype0

"quant_conv2d_1/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_1/post_activation_max

6quant_conv2d_1/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_1/post_activation_max*
_output_shapes
: *
dtype0

"quant_conv2d_1/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_1/post_activation_min

6quant_conv2d_1/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_1/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d_1/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_1/kernel_max

-quant_conv2d_1/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_1/kernel_max*
_output_shapes
:*
dtype0

quant_conv2d_1/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_1/kernel_min

-quant_conv2d_1/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_1/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_1/optimizer_step

1quant_conv2d_1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_1/optimizer_step*
_output_shapes
: *
dtype0

 quant_conv2d/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_conv2d/post_activation_max

4quant_conv2d/post_activation_max/Read/ReadVariableOpReadVariableOp quant_conv2d/post_activation_max*
_output_shapes
: *
dtype0

 quant_conv2d/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_conv2d/post_activation_min

4quant_conv2d/post_activation_min/Read/ReadVariableOpReadVariableOp quant_conv2d/post_activation_min*
_output_shapes
: *
dtype0

quant_conv2d/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namequant_conv2d/kernel_max

+quant_conv2d/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d/kernel_max*
_output_shapes
:*
dtype0

quant_conv2d/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namequant_conv2d/kernel_min

+quant_conv2d/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namequant_conv2d/optimizer_step

/quant_conv2d/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d/optimizer_step*
_output_shapes
: *
dtype0

quantize_layer/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequantize_layer/optimizer_step

1quantize_layer/optimizer_step/Read/ReadVariableOpReadVariableOpquantize_layer/optimizer_step*
_output_shapes
: *
dtype0

!quantize_layer/quantize_layer_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_max

5quantize_layer/quantize_layer_max/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_max*
_output_shapes
: *
dtype0

!quantize_layer/quantize_layer_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_min

5quantize_layer/quantize_layer_min/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_min*
_output_shapes
: *
dtype0

NoOpNoOp
®
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ï­
valueÄ­BÀ­ B¸­
Ó
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer-11
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
è
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
quantize_layer_min
quantize_layer_max
quantizer_vars
optimizer_step*
Æ
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
	&layer
'optimizer_step
(_weight_vars
)
kernel_min
*
kernel_max
+_quantize_activations
,post_activation_min
-post_activation_max
._output_quantizers*
Æ
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
	5layer
6optimizer_step
7_weight_vars
8
kernel_min
9
kernel_max
:_quantize_activations
;post_activation_min
<post_activation_max
=_output_quantizers*
Æ
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
	Dlayer
Eoptimizer_step
F_weight_vars
G
kernel_min
H
kernel_max
I_quantize_activations
Jpost_activation_min
Kpost_activation_max
L_output_quantizers*
Æ
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
	Slayer
Toptimizer_step
U_weight_vars
V
kernel_min
W
kernel_max
X_quantize_activations
Ypost_activation_min
Zpost_activation_max
[_output_quantizers*
Æ
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
	blayer
coptimizer_step
d_weight_vars
e
kernel_min
f
kernel_max
g_quantize_activations
hpost_activation_min
ipost_activation_max
j_output_quantizers*
Æ
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
	qlayer
roptimizer_step
s_weight_vars
t
kernel_min
u
kernel_max
v_quantize_activations
wpost_activation_min
xpost_activation_max
y_output_quantizers*

z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
Õ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

layer
optimizer_step
_weight_vars

kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers*
¾
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers

output_min

output_max
_output_quantizer_vars*

	variables
trainable_variables
regularization_losses
 	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses* 

£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api
§__call__
+¨&call_and_return_all_conditional_losses* 
È
0
1
2
©3
ª4
'5
)6
*7
,8
-9
«10
¬11
612
813
914
;15
<16
­17
®18
E19
G20
H21
J22
K23
¯24
°25
T26
V27
W28
Y29
Z30
±31
²32
c33
e34
f35
h36
i37
³38
´39
r40
t41
u42
w43
x44
µ45
¶46
47
48
49
50
51
52
53
54*
x
©0
ª1
«2
¬3
­4
®5
¯6
°7
±8
²9
³10
´11
µ12
¶13*
* 
µ
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
¼trace_0
½trace_1
¾trace_2
¿trace_3* 
:
Àtrace_0
Átrace_1
Âtrace_2
Ãtrace_3* 
* 

Äserving_default* 

0
1
2*
* 
* 

Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Êtrace_0
Ëtrace_1* 

Ìtrace_0
Ítrace_1* 
}w
VARIABLE_VALUE!quantize_layer/quantize_layer_minBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!quantize_layer/quantize_layer_maxBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUE*

min_var
max_var*
uo
VARIABLE_VALUEquantize_layer/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
7
©0
ª1
'2
)3
*4
,5
-6*

©0
ª1*
* 

Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

Ótrace_0
Ôtrace_1* 

Õtrace_0
Ötrace_1* 
Ñ
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses
©kernel
	ªbias
!Ý_jit_compiled_convolution_op*
sm
VARIABLE_VALUEquant_conv2d/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

Þ0*
ke
VARIABLE_VALUEquant_conv2d/kernel_min:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEquant_conv2d/kernel_max:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
}w
VARIABLE_VALUE quant_conv2d/post_activation_minClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE quant_conv2d/post_activation_maxClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
7
«0
¬1
62
83
94
;5
<6*

«0
¬1*
* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

ätrace_0
åtrace_1* 

ætrace_0
çtrace_1* 
Ñ
è	variables
étrainable_variables
êregularization_losses
ë	keras_api
ì__call__
+í&call_and_return_all_conditional_losses
«kernel
	¬bias
!î_jit_compiled_convolution_op*
uo
VARIABLE_VALUEquant_conv2d_1/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

ï0*
mg
VARIABLE_VALUEquant_conv2d_1/kernel_min:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_conv2d_1/kernel_max:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUE"quant_conv2d_1/post_activation_minClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"quant_conv2d_1/post_activation_maxClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
7
­0
®1
E2
G3
H4
J5
K6*

­0
®1*
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

õtrace_0
ötrace_1* 

÷trace_0
øtrace_1* 
Ñ
ù	variables
útrainable_variables
ûregularization_losses
ü	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses
­kernel
	®bias
!ÿ_jit_compiled_convolution_op*
uo
VARIABLE_VALUEquant_conv2d_2/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

0*
mg
VARIABLE_VALUEquant_conv2d_2/kernel_min:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_conv2d_2/kernel_max:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUE"quant_conv2d_2/post_activation_minClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"quant_conv2d_2/post_activation_maxClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
7
¯0
°1
T2
V3
W4
Y5
Z6*

¯0
°1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
Ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
¯kernel
	°bias
!_jit_compiled_convolution_op*
uo
VARIABLE_VALUEquant_conv2d_3/optimizer_step>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

0*
mg
VARIABLE_VALUEquant_conv2d_3/kernel_min:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_conv2d_3/kernel_max:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUE"quant_conv2d_3/post_activation_minClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"quant_conv2d_3/post_activation_maxClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
7
±0
²1
c2
e3
f4
h5
i6*

±0
²1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
Ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
±kernel
	²bias
!¡_jit_compiled_convolution_op*
uo
VARIABLE_VALUEquant_conv2d_4/optimizer_step>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

¢0*
mg
VARIABLE_VALUEquant_conv2d_4/kernel_min:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_conv2d_4/kernel_max:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUE"quant_conv2d_4/post_activation_minClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"quant_conv2d_4/post_activation_maxClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
7
³0
´1
r2
t3
u4
w5
x6*

³0
´1*
* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

¨trace_0
©trace_1* 

ªtrace_0
«trace_1* 
Ñ
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses
³kernel
	´bias
!²_jit_compiled_convolution_op*
uo
VARIABLE_VALUEquant_conv2d_5/optimizer_step>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

³0*
mg
VARIABLE_VALUEquant_conv2d_5/kernel_min:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_conv2d_5/kernel_max:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUE"quant_conv2d_5/post_activation_minClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"quant_conv2d_5/post_activation_maxClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

¹trace_0
ºtrace_1* 

»trace_0
¼trace_1* 
<
µ0
¶1
2
3
4
5
6*

µ0
¶1*
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Âtrace_0
Ãtrace_1* 

Ätrace_0
Åtrace_1* 
Ñ
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses
µkernel
	¶bias
!Ì_jit_compiled_convolution_op*
uo
VARIABLE_VALUEquant_conv2d_6/optimizer_step>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

Í0*
mg
VARIABLE_VALUEquant_conv2d_6/kernel_min:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEquant_conv2d_6/kernel_max:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUE"quant_conv2d_6/post_activation_minClayer_with_weights-7/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"quant_conv2d_6/post_activation_maxClayer_with_weights-7/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*
* 
* 

Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ótrace_0
Ôtrace_1* 

Õtrace_0
Ötrace_1* 

×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses* 
pj
VARIABLE_VALUEquant_add/optimizer_step>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
hb
VARIABLE_VALUEquant_add/output_min:layer_with_weights-8/output_min/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEquant_add/output_max:layer_with_weights-8/output_max/.ATTRIBUTES/VARIABLE_VALUE*
 
min_var
max_var*
* 
* 
* 

Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
	variables
trainable_variables
regularization_losses
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses* 

âtrace_0
ãtrace_1* 

ätrace_0
åtrace_1* 
* 
* 
* 

ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
£	variables
¤trainable_variables
¥regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses* 

ëtrace_0
ìtrace_1* 

ítrace_0
îtrace_1* 
MG
VARIABLE_VALUEconv2d/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_1/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_1/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_2/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_2/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_4/kernel'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_4/bias'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_5/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_5/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_6/kernel'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_6/bias'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
Ê
0
1
2
'3
)4
*5
,6
-7
68
89
910
;11
<12
E13
G14
H15
J16
K17
T18
V19
W20
Y21
Z22
c23
e24
f25
h26
i27
r28
t29
u30
w31
x32
33
34
35
36
37
38
39
40*
b
0
1
2
3
4
5
6
7
	8

9
10
11
12*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1
2*
* 
* 
* 
* 
* 
* 
* 
* 
'
'0
)1
*2
,3
-4*

&0*
* 
* 
* 
* 
* 
* 
* 

ª0*

ª0*
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses*
* 
* 
* 

©0
ô2*
'
60
81
92
;3
<4*

50*
* 
* 
* 
* 
* 
* 
* 

¬0*

¬0*
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
è	variables
étrainable_variables
êregularization_losses
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses*
* 
* 
* 

«0
ú2*
'
E0
G1
H2
J3
K4*

D0*
* 
* 
* 
* 
* 
* 
* 

®0*

®0*
* 

ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
ù	variables
útrainable_variables
ûregularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses*
* 
* 
* 

­0
2*
'
T0
V1
W2
Y3
Z4*

S0*
* 
* 
* 
* 
* 
* 
* 

°0*

°0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 

¯0
2*
'
c0
e1
f2
h3
i4*

b0*
* 
* 
* 
* 
* 
* 
* 

²0*

²0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 
* 

±0
2*
'
r0
t1
u2
w3
x4*

q0*
* 
* 
* 
* 
* 
* 
* 

´0*

´0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses*
* 
* 
* 

³0
2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
,
0
1
2
3
4*

0*
* 
* 
* 
* 
* 
* 
* 

¶0*

¶0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses*
* 
* 
* 

µ0
2*

0
1
2*


0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

)min_var
*max_var*
* 
* 
* 
* 
* 

8min_var
9max_var*
* 
* 
* 
* 
* 

Gmin_var
Hmax_var*
* 
* 
* 
* 
* 

Vmin_var
Wmax_var*
* 
* 
* 
* 
* 

emin_var
fmax_var*
* 
* 
* 
* 
* 

tmin_var
umax_var*
* 
* 
* 
* 
* 
 
min_var
max_var*
* 
* 
* 
* 
* 
®
serving_default_input_1Placeholder*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*6
shape-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Þ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxconv2d/kernelquant_conv2d/kernel_minquant_conv2d/kernel_maxconv2d/bias quant_conv2d/post_activation_min quant_conv2d/post_activation_maxconv2d_1/kernelquant_conv2d_1/kernel_minquant_conv2d_1/kernel_maxconv2d_1/bias"quant_conv2d_1/post_activation_min"quant_conv2d_1/post_activation_maxconv2d_2/kernelquant_conv2d_2/kernel_minquant_conv2d_2/kernel_maxconv2d_2/bias"quant_conv2d_2/post_activation_min"quant_conv2d_2/post_activation_maxconv2d_3/kernelquant_conv2d_3/kernel_minquant_conv2d_3/kernel_maxconv2d_3/bias"quant_conv2d_3/post_activation_min"quant_conv2d_3/post_activation_maxconv2d_4/kernelquant_conv2d_4/kernel_minquant_conv2d_4/kernel_maxconv2d_4/bias"quant_conv2d_4/post_activation_min"quant_conv2d_4/post_activation_maxconv2d_5/kernelquant_conv2d_5/kernel_minquant_conv2d_5/kernel_maxconv2d_5/bias"quant_conv2d_5/post_activation_min"quant_conv2d_5/post_activation_maxconv2d_6/kernelquant_conv2d_6/kernel_minquant_conv2d_6/kernel_maxconv2d_6/bias"quant_conv2d_6/post_activation_min"quant_conv2d_6/post_activation_maxquant_add/output_minquant_add/output_max*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_5657
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ä
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5quantize_layer/quantize_layer_min/Read/ReadVariableOp5quantize_layer/quantize_layer_max/Read/ReadVariableOp1quantize_layer/optimizer_step/Read/ReadVariableOp/quant_conv2d/optimizer_step/Read/ReadVariableOp+quant_conv2d/kernel_min/Read/ReadVariableOp+quant_conv2d/kernel_max/Read/ReadVariableOp4quant_conv2d/post_activation_min/Read/ReadVariableOp4quant_conv2d/post_activation_max/Read/ReadVariableOp1quant_conv2d_1/optimizer_step/Read/ReadVariableOp-quant_conv2d_1/kernel_min/Read/ReadVariableOp-quant_conv2d_1/kernel_max/Read/ReadVariableOp6quant_conv2d_1/post_activation_min/Read/ReadVariableOp6quant_conv2d_1/post_activation_max/Read/ReadVariableOp1quant_conv2d_2/optimizer_step/Read/ReadVariableOp-quant_conv2d_2/kernel_min/Read/ReadVariableOp-quant_conv2d_2/kernel_max/Read/ReadVariableOp6quant_conv2d_2/post_activation_min/Read/ReadVariableOp6quant_conv2d_2/post_activation_max/Read/ReadVariableOp1quant_conv2d_3/optimizer_step/Read/ReadVariableOp-quant_conv2d_3/kernel_min/Read/ReadVariableOp-quant_conv2d_3/kernel_max/Read/ReadVariableOp6quant_conv2d_3/post_activation_min/Read/ReadVariableOp6quant_conv2d_3/post_activation_max/Read/ReadVariableOp1quant_conv2d_4/optimizer_step/Read/ReadVariableOp-quant_conv2d_4/kernel_min/Read/ReadVariableOp-quant_conv2d_4/kernel_max/Read/ReadVariableOp6quant_conv2d_4/post_activation_min/Read/ReadVariableOp6quant_conv2d_4/post_activation_max/Read/ReadVariableOp1quant_conv2d_5/optimizer_step/Read/ReadVariableOp-quant_conv2d_5/kernel_min/Read/ReadVariableOp-quant_conv2d_5/kernel_max/Read/ReadVariableOp6quant_conv2d_5/post_activation_min/Read/ReadVariableOp6quant_conv2d_5/post_activation_max/Read/ReadVariableOp1quant_conv2d_6/optimizer_step/Read/ReadVariableOp-quant_conv2d_6/kernel_min/Read/ReadVariableOp-quant_conv2d_6/kernel_max/Read/ReadVariableOp6quant_conv2d_6/post_activation_min/Read/ReadVariableOp6quant_conv2d_6/post_activation_max/Read/ReadVariableOp,quant_add/optimizer_step/Read/ReadVariableOp(quant_add/output_min/Read/ReadVariableOp(quant_add/output_max/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOpConst*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_7455

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxquantize_layer/optimizer_stepquant_conv2d/optimizer_stepquant_conv2d/kernel_minquant_conv2d/kernel_max quant_conv2d/post_activation_min quant_conv2d/post_activation_maxquant_conv2d_1/optimizer_stepquant_conv2d_1/kernel_minquant_conv2d_1/kernel_max"quant_conv2d_1/post_activation_min"quant_conv2d_1/post_activation_maxquant_conv2d_2/optimizer_stepquant_conv2d_2/kernel_minquant_conv2d_2/kernel_max"quant_conv2d_2/post_activation_min"quant_conv2d_2/post_activation_maxquant_conv2d_3/optimizer_stepquant_conv2d_3/kernel_minquant_conv2d_3/kernel_max"quant_conv2d_3/post_activation_min"quant_conv2d_3/post_activation_maxquant_conv2d_4/optimizer_stepquant_conv2d_4/kernel_minquant_conv2d_4/kernel_max"quant_conv2d_4/post_activation_min"quant_conv2d_4/post_activation_maxquant_conv2d_5/optimizer_stepquant_conv2d_5/kernel_minquant_conv2d_5/kernel_max"quant_conv2d_5/post_activation_min"quant_conv2d_5/post_activation_maxquant_conv2d_6/optimizer_stepquant_conv2d_6/kernel_minquant_conv2d_6/kernel_max"quant_conv2d_6/post_activation_min"quant_conv2d_6/post_activation_maxquant_add/optimizer_stepquant_add/output_minquant_add/output_maxconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/bias*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_7630"
Ò	

(__inference_quant_add_layer_call_fn_7183
inputs_0
inputs_1
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_quant_add_layer_call_and_return_conditional_losses_4283
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ó	
þ
-__inference_quant_conv2d_2_layer_call_fn_6628

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_3913
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²	

@__inference_lambda_layer_call_and_return_conditional_losses_4041

inputs
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
concatConcatV2inputsinputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*É
_input_shapes·
´:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
^
B__inference_lambda_1_layer_call_and_return_conditional_losses_7236

inputs
identity
DepthToSpaceDepthToSpaceinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*

block_sizew
IdentityIdentityDepthToSpace:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


m
%__inference_lambda_layer_call_fn_7035
inputs_0
inputs_1
inputs_2
inputs_3
identityè
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_4041z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*É
_input_shapes·
´:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3
Ï	
þ
-__inference_quant_conv2d_5_layer_call_fn_6957

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_4468
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³


"__inference_signature_wrapper_5657
input_1
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: #
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: $

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: $

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: $

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: $

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35: 

unknown_36: $

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41: 

unknown_42: 

unknown_43: 

unknown_44: 
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_3800
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ë	
ü
+__inference_quant_conv2d_layer_call_fn_6437

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_4888
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾	

@__inference_lambda_layer_call_and_return_conditional_losses_7052
inputs_0
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¥
concatConcatV2inputs_0inputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*É
_input_shapes·
´:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3
£

-__inference_quantize_layer_layer_call_fn_6373

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantize_layer_layer_call_and_return_conditional_losses_4936
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
û
H__inference_quantize_layer_layer_call_and_return_conditional_losses_6382

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1²
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
S
	
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_4468

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


m
%__inference_lambda_layer_call_fn_7043
inputs_0
inputs_1
inputs_2
inputs_3
identityè
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_4396z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*É
_input_shapes·
´:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3
¡K

>__inference_abpn_layer_call_and_return_conditional_losses_5150

inputs
quantize_layer_5045: 
quantize_layer_5047: +
quant_conv2d_5050:
quant_conv2d_5052:
quant_conv2d_5054:
quant_conv2d_5056:
quant_conv2d_5058: 
quant_conv2d_5060: -
quant_conv2d_1_5063:!
quant_conv2d_1_5065:!
quant_conv2d_1_5067:!
quant_conv2d_1_5069:
quant_conv2d_1_5071: 
quant_conv2d_1_5073: -
quant_conv2d_2_5076:!
quant_conv2d_2_5078:!
quant_conv2d_2_5080:!
quant_conv2d_2_5082:
quant_conv2d_2_5084: 
quant_conv2d_2_5086: -
quant_conv2d_3_5089:!
quant_conv2d_3_5091:!
quant_conv2d_3_5093:!
quant_conv2d_3_5095:
quant_conv2d_3_5097: 
quant_conv2d_3_5099: -
quant_conv2d_4_5102:!
quant_conv2d_4_5104:!
quant_conv2d_4_5106:!
quant_conv2d_4_5108:
quant_conv2d_4_5110: 
quant_conv2d_4_5112: -
quant_conv2d_5_5115:!
quant_conv2d_5_5117:!
quant_conv2d_5_5119:!
quant_conv2d_5_5121:
quant_conv2d_5_5123: 
quant_conv2d_5_5125: -
quant_conv2d_6_5129:!
quant_conv2d_6_5131:!
quant_conv2d_6_5133:!
quant_conv2d_6_5135:
quant_conv2d_6_5137: 
quant_conv2d_6_5139: 
quant_add_5142: 
quant_add_5144: 
identity¢!quant_add/StatefulPartitionedCall¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢&quant_conv2d_3/StatefulPartitionedCall¢&quant_conv2d_4/StatefulPartitionedCall¢&quant_conv2d_5/StatefulPartitionedCall¢&quant_conv2d_6/StatefulPartitionedCall¢&quantize_layer/StatefulPartitionedCall
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_5045quantize_layer_5047*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantize_layer_layer_call_and_return_conditional_losses_4936
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_5050quant_conv2d_5052quant_conv2d_5054quant_conv2d_5056quant_conv2d_5058quant_conv2d_5060*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_4888
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0quant_conv2d_1_5063quant_conv2d_1_5065quant_conv2d_1_5067quant_conv2d_1_5069quant_conv2d_1_5071quant_conv2d_1_5073*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_4804
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0quant_conv2d_2_5076quant_conv2d_2_5078quant_conv2d_2_5080quant_conv2d_2_5082quant_conv2d_2_5084quant_conv2d_2_5086*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_4720
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_5089quant_conv2d_3_5091quant_conv2d_3_5093quant_conv2d_3_5095quant_conv2d_3_5097quant_conv2d_3_5099*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_4636
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_5102quant_conv2d_4_5104quant_conv2d_4_5106quant_conv2d_4_5108quant_conv2d_4_5110quant_conv2d_4_5112*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_4552
&quant_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0quant_conv2d_5_5115quant_conv2d_5_5117quant_conv2d_5_5119quant_conv2d_5_5121quant_conv2d_5_5123quant_conv2d_5_5125*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_4468
lambda/PartitionedCallPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_4396
&quant_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_5/StatefulPartitionedCall:output:0quant_conv2d_6_5129quant_conv2d_6_5131quant_conv2d_6_5133quant_conv2d_6_5135quant_conv2d_6_5137quant_conv2d_6_5139*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_4358Ï
!quant_add/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0/quant_conv2d_6/StatefulPartitionedCall:output:0quant_add_5142quant_add_5144*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_quant_add_layer_call_and_return_conditional_losses_4283ô
lambda_1/PartitionedCallPartitionedCall*quant_add/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_1_layer_call_and_return_conditional_losses_4239ë
lambda_2/PartitionedCallPartitionedCall!lambda_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_2_layer_call_and_return_conditional_losses_4223
IdentityIdentity!lambda_2/PartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp"^quant_add/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall'^quant_conv2d_5/StatefulPartitionedCall'^quant_conv2d_6/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_add/StatefulPartitionedCall!quant_add/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2P
&quant_conv2d_5/StatefulPartitionedCall&quant_conv2d_5/StatefulPartitionedCall2P
&quant_conv2d_6/StatefulPartitionedCall&quant_conv2d_6/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

-__inference_quantize_layer_layer_call_fn_6364

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantize_layer_layer_call_and_return_conditional_losses_3816
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
S
	
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_6819

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø#
û
H__inference_quantize_layer_layer_call_and_return_conditional_losses_6403

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity¢#AllValuesQuantize/AssignMaxAllValue¢#AllValuesQuantize/AssignMinAllValue¢8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢(AllValuesQuantize/Maximum/ReadVariableOp¢(AllValuesQuantize/Minimum/ReadVariableOpp
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: r
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: ï
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ï
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(È
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0Ê
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
S
	
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_7027

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°


#__inference_abpn_layer_call_fn_5851

inputs
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: #
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: $

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: $

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: $

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: $

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35: 

unknown_36: $

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41: 

unknown_42: 

unknown_43: 

unknown_44: 
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	!$'**-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_abpn_layer_call_and_return_conditional_losses_5150
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó	
þ
-__inference_quant_conv2d_4_layer_call_fn_6836

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_3983
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
^
B__inference_lambda_1_layer_call_and_return_conditional_losses_4239

inputs
identity
DepthToSpaceDepthToSpaceinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*

block_sizew
IdentityIdentityDepthToSpace:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
^
B__inference_lambda_1_layer_call_and_return_conditional_losses_7241

inputs
identity
DepthToSpaceDepthToSpaceinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*

block_sizew
IdentityIdentityDepthToSpace:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

C
'__inference_lambda_2_layer_call_fn_7251

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_2_layer_call_and_return_conditional_losses_4223z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
S
	
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_6507

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ$
¥
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_6874

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦R
	
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_7163

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤K

>__inference_abpn_layer_call_and_return_conditional_losses_5558
input_1
quantize_layer_5453: 
quantize_layer_5455: +
quant_conv2d_5458:
quant_conv2d_5460:
quant_conv2d_5462:
quant_conv2d_5464:
quant_conv2d_5466: 
quant_conv2d_5468: -
quant_conv2d_1_5471:!
quant_conv2d_1_5473:!
quant_conv2d_1_5475:!
quant_conv2d_1_5477:
quant_conv2d_1_5479: 
quant_conv2d_1_5481: -
quant_conv2d_2_5484:!
quant_conv2d_2_5486:!
quant_conv2d_2_5488:!
quant_conv2d_2_5490:
quant_conv2d_2_5492: 
quant_conv2d_2_5494: -
quant_conv2d_3_5497:!
quant_conv2d_3_5499:!
quant_conv2d_3_5501:!
quant_conv2d_3_5503:
quant_conv2d_3_5505: 
quant_conv2d_3_5507: -
quant_conv2d_4_5510:!
quant_conv2d_4_5512:!
quant_conv2d_4_5514:!
quant_conv2d_4_5516:
quant_conv2d_4_5518: 
quant_conv2d_4_5520: -
quant_conv2d_5_5523:!
quant_conv2d_5_5525:!
quant_conv2d_5_5527:!
quant_conv2d_5_5529:
quant_conv2d_5_5531: 
quant_conv2d_5_5533: -
quant_conv2d_6_5537:!
quant_conv2d_6_5539:!
quant_conv2d_6_5541:!
quant_conv2d_6_5543:
quant_conv2d_6_5545: 
quant_conv2d_6_5547: 
quant_add_5550: 
quant_add_5552: 
identity¢!quant_add/StatefulPartitionedCall¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢&quant_conv2d_3/StatefulPartitionedCall¢&quant_conv2d_4/StatefulPartitionedCall¢&quant_conv2d_5/StatefulPartitionedCall¢&quant_conv2d_6/StatefulPartitionedCall¢&quantize_layer/StatefulPartitionedCall
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1quantize_layer_5453quantize_layer_5455*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantize_layer_layer_call_and_return_conditional_losses_4936
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_5458quant_conv2d_5460quant_conv2d_5462quant_conv2d_5464quant_conv2d_5466quant_conv2d_5468*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_4888
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0quant_conv2d_1_5471quant_conv2d_1_5473quant_conv2d_1_5475quant_conv2d_1_5477quant_conv2d_1_5479quant_conv2d_1_5481*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_4804
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0quant_conv2d_2_5484quant_conv2d_2_5486quant_conv2d_2_5488quant_conv2d_2_5490quant_conv2d_2_5492quant_conv2d_2_5494*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_4720
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_5497quant_conv2d_3_5499quant_conv2d_3_5501quant_conv2d_3_5503quant_conv2d_3_5505quant_conv2d_3_5507*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_4636
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_5510quant_conv2d_4_5512quant_conv2d_4_5514quant_conv2d_4_5516quant_conv2d_4_5518quant_conv2d_4_5520*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_4552
&quant_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0quant_conv2d_5_5523quant_conv2d_5_5525quant_conv2d_5_5527quant_conv2d_5_5529quant_conv2d_5_5531quant_conv2d_5_5533*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_4468
lambda/PartitionedCallPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_4396
&quant_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_5/StatefulPartitionedCall:output:0quant_conv2d_6_5537quant_conv2d_6_5539quant_conv2d_6_5541quant_conv2d_6_5543quant_conv2d_6_5545quant_conv2d_6_5547*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_4358Ï
!quant_add/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0/quant_conv2d_6/StatefulPartitionedCall:output:0quant_add_5550quant_add_5552*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_quant_add_layer_call_and_return_conditional_losses_4283ô
lambda_1/PartitionedCallPartitionedCall*quant_add/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_1_layer_call_and_return_conditional_losses_4239ë
lambda_2/PartitionedCallPartitionedCall!lambda_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_2_layer_call_and_return_conditional_losses_4223
IdentityIdentity!lambda_2/PartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp"^quant_add/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall'^quant_conv2d_5/StatefulPartitionedCall'^quant_conv2d_6/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_add/StatefulPartitionedCall!quant_add/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2P
&quant_conv2d_5/StatefulPartitionedCall&quant_conv2d_5/StatefulPartitionedCall2P
&quant_conv2d_6/StatefulPartitionedCall&quant_conv2d_6/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:j f
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

C
'__inference_lambda_1_layer_call_fn_7226

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_1_layer_call_and_return_conditional_losses_4099z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
åÉ
@
__inference__wrapped_model_3800
input_1_
Uabpn_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: a
Wabpn_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: t
Zabpn_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:j
\abpn_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:j
\abpn_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:?
1abpn_quant_conv2d_biasadd_readvariableop_resource:]
Sabpn_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: _
Uabpn_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: v
\abpn_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:l
^abpn_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:l
^abpn_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:A
3abpn_quant_conv2d_1_biasadd_readvariableop_resource:_
Uabpn_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: a
Wabpn_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: v
\abpn_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:l
^abpn_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:l
^abpn_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:A
3abpn_quant_conv2d_2_biasadd_readvariableop_resource:_
Uabpn_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: a
Wabpn_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: v
\abpn_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:l
^abpn_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:l
^abpn_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:A
3abpn_quant_conv2d_3_biasadd_readvariableop_resource:_
Uabpn_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: a
Wabpn_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: v
\abpn_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:l
^abpn_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:l
^abpn_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:A
3abpn_quant_conv2d_4_biasadd_readvariableop_resource:_
Uabpn_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: a
Wabpn_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: v
\abpn_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:l
^abpn_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:l
^abpn_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:A
3abpn_quant_conv2d_5_biasadd_readvariableop_resource:_
Uabpn_quant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: a
Wabpn_quant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: v
\abpn_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:l
^abpn_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:l
^abpn_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:A
3abpn_quant_conv2d_6_biasadd_readvariableop_resource:_
Uabpn_quant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: a
Wabpn_quant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: Z
Pabpn_quant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rabpn_quant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢Gabpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iabpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢(abpn/quant_conv2d/BiasAdd/ReadVariableOp¢Qabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Sabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Sabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Jabpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Labpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢*abpn/quant_conv2d_1/BiasAdd/ReadVariableOp¢Sabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Uabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Uabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Labpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Nabpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢*abpn/quant_conv2d_2/BiasAdd/ReadVariableOp¢Sabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Uabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Uabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Labpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Nabpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢*abpn/quant_conv2d_3/BiasAdd/ReadVariableOp¢Sabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Uabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Uabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Labpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Nabpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢*abpn/quant_conv2d_4/BiasAdd/ReadVariableOp¢Sabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Uabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Uabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Labpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Nabpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢*abpn/quant_conv2d_5/BiasAdd/ReadVariableOp¢Sabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Uabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Uabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Labpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Nabpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢*abpn/quant_conv2d_6/BiasAdd/ReadVariableOp¢Sabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Uabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Uabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Labpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Nabpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Labpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Nabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ú
Labpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpUabpn_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Þ
Nabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpWabpn_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ò
=abpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinput_1Tabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Vabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
Qabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpZabpn_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0ì
Sabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp\abpn_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0ì
Sabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp\abpn_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0¸
Babpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelYabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0[abpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0[abpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(­
abpn/quant_conv2d/Conv2DConv2DGabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Labpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

(abpn/quant_conv2d/BiasAdd/ReadVariableOpReadVariableOp1abpn_quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
abpn/quant_conv2d/BiasAddBiasAdd!abpn/quant_conv2d/Conv2D:output:00abpn/quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
abpn/quant_conv2d/ReluRelu"abpn/quant_conv2d/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÖ
Jabpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpSabpn_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ú
Labpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpUabpn_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0é
;abpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars$abpn/quant_conv2d/Relu:activations:0Rabpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Tabpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿø
Sabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp\abpn_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp^abpn_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp^abpn_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0À
Dabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel[abpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0]abpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0]abpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(¯
abpn/quant_conv2d_1/Conv2DConv2DEabpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Nabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

*abpn/quant_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3abpn_quant_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ë
abpn/quant_conv2d_1/BiasAddBiasAdd#abpn/quant_conv2d_1/Conv2D:output:02abpn/quant_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
abpn/quant_conv2d_1/ReluRelu$abpn/quant_conv2d_1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
Labpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpUabpn_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Þ
Nabpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpWabpn_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0ñ
=abpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars&abpn/quant_conv2d_1/Relu:activations:0Tabpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Vabpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿø
Sabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp\abpn_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp^abpn_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp^abpn_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0À
Dabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel[abpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0]abpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0]abpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(±
abpn/quant_conv2d_2/Conv2DConv2DGabpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Nabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

*abpn/quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3abpn_quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ë
abpn/quant_conv2d_2/BiasAddBiasAdd#abpn/quant_conv2d_2/Conv2D:output:02abpn/quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
abpn/quant_conv2d_2/ReluRelu$abpn/quant_conv2d_2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
Labpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpUabpn_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Þ
Nabpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpWabpn_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0ñ
=abpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars&abpn/quant_conv2d_2/Relu:activations:0Tabpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Vabpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿø
Sabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp\abpn_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp^abpn_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp^abpn_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0À
Dabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel[abpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0]abpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0]abpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(±
abpn/quant_conv2d_3/Conv2DConv2DGabpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Nabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

*abpn/quant_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3abpn_quant_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ë
abpn/quant_conv2d_3/BiasAddBiasAdd#abpn/quant_conv2d_3/Conv2D:output:02abpn/quant_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
abpn/quant_conv2d_3/ReluRelu$abpn/quant_conv2d_3/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
Labpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpUabpn_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Þ
Nabpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpWabpn_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0ñ
=abpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars&abpn/quant_conv2d_3/Relu:activations:0Tabpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Vabpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿø
Sabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp\abpn_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp^abpn_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp^abpn_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0À
Dabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel[abpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0]abpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0]abpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(±
abpn/quant_conv2d_4/Conv2DConv2DGabpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Nabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

*abpn/quant_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp3abpn_quant_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ë
abpn/quant_conv2d_4/BiasAddBiasAdd#abpn/quant_conv2d_4/Conv2D:output:02abpn/quant_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
abpn/quant_conv2d_4/ReluRelu$abpn/quant_conv2d_4/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
Labpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpUabpn_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Þ
Nabpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpWabpn_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0ñ
=abpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars&abpn/quant_conv2d_4/Relu:activations:0Tabpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Vabpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿø
Sabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp\abpn_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp^abpn_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp^abpn_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0À
Dabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel[abpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0]abpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0]abpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(±
abpn/quant_conv2d_5/Conv2DConv2DGabpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Nabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

*abpn/quant_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp3abpn_quant_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ë
abpn/quant_conv2d_5/BiasAddBiasAdd#abpn/quant_conv2d_5/Conv2D:output:02abpn/quant_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
abpn/quant_conv2d_5/ReluRelu$abpn/quant_conv2d_5/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
Labpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpUabpn_quant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Þ
Nabpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpWabpn_quant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0ñ
=abpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars&abpn/quant_conv2d_5/Relu:activations:0Tabpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Vabpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿY
abpn/lambda/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¹
abpn/lambda/concatConcatV2Gabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Gabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Gabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Gabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0 abpn/lambda/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿø
Sabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp\abpn_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp^abpn_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0ð
Uabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp^abpn_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0À
Dabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel[abpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0]abpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0]abpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(±
abpn/quant_conv2d_6/Conv2DConv2DGabpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Nabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

*abpn/quant_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp3abpn_quant_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ë
abpn/quant_conv2d_6/BiasAddBiasAdd#abpn/quant_conv2d_6/Conv2D:output:02abpn/quant_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
Labpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpUabpn_quant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Þ
Nabpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpWabpn_quant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0ï
=abpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars$abpn/quant_conv2d_6/BiasAdd:output:0Tabpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Vabpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÍ
abpn/quant_add/addAddV2abpn/lambda/concat:output:0Gabpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
Gabpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPabpn_quant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iabpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRabpn_quant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ò
8abpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsabpn/quant_add/add:z:0Oabpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qabpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÌ
abpn/lambda_1/DepthToSpaceDepthToSpaceBabpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*

block_sizej
%abpn/lambda_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CÏ
#abpn/lambda_2/clip_by_value/MinimumMinimum#abpn/lambda_1/DepthToSpace:output:0.abpn/lambda_2/clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿb
abpn/lambda_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
abpn/lambda_2/clip_by_valueMaximum'abpn/lambda_2/clip_by_value/Minimum:z:0&abpn/lambda_2/clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentityabpn/lambda_2/clip_by_value:z:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ
NoOpNoOpH^abpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^abpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^abpn/quant_conv2d/BiasAdd/ReadVariableOpR^abpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpT^abpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1T^abpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2K^abpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpM^abpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1+^abpn/quant_conv2d_1/BiasAdd/ReadVariableOpT^abpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpV^abpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1V^abpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2M^abpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpO^abpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1+^abpn/quant_conv2d_2/BiasAdd/ReadVariableOpT^abpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpV^abpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1V^abpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2M^abpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpO^abpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1+^abpn/quant_conv2d_3/BiasAdd/ReadVariableOpT^abpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpV^abpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1V^abpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2M^abpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpO^abpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1+^abpn/quant_conv2d_4/BiasAdd/ReadVariableOpT^abpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpV^abpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1V^abpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2M^abpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpO^abpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1+^abpn/quant_conv2d_5/BiasAdd/ReadVariableOpT^abpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpV^abpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1V^abpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2M^abpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpO^abpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1+^abpn/quant_conv2d_6/BiasAdd/ReadVariableOpT^abpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpV^abpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1V^abpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2M^abpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpO^abpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1M^abpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpO^abpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Gabpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGabpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iabpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iabpn/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(abpn/quant_conv2d/BiasAdd/ReadVariableOp(abpn/quant_conv2d/BiasAdd/ReadVariableOp2¦
Qabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2ª
Sabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Sabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12ª
Sabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Sabpn/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Jabpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJabpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Labpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Labpn/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12X
*abpn/quant_conv2d_1/BiasAdd/ReadVariableOp*abpn/quant_conv2d_1/BiasAdd/ReadVariableOp2ª
Sabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpSabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2®
Uabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Uabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12®
Uabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Uabpn/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Labpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpLabpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2 
Nabpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Nabpn/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12X
*abpn/quant_conv2d_2/BiasAdd/ReadVariableOp*abpn/quant_conv2d_2/BiasAdd/ReadVariableOp2ª
Sabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpSabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2®
Uabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Uabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12®
Uabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Uabpn/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Labpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpLabpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2 
Nabpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Nabpn/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12X
*abpn/quant_conv2d_3/BiasAdd/ReadVariableOp*abpn/quant_conv2d_3/BiasAdd/ReadVariableOp2ª
Sabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpSabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2®
Uabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Uabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12®
Uabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Uabpn/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Labpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpLabpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2 
Nabpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Nabpn/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12X
*abpn/quant_conv2d_4/BiasAdd/ReadVariableOp*abpn/quant_conv2d_4/BiasAdd/ReadVariableOp2ª
Sabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpSabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2®
Uabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Uabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12®
Uabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Uabpn/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Labpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpLabpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2 
Nabpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Nabpn/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12X
*abpn/quant_conv2d_5/BiasAdd/ReadVariableOp*abpn/quant_conv2d_5/BiasAdd/ReadVariableOp2ª
Sabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpSabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2®
Uabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Uabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12®
Uabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Uabpn/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Labpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpLabpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2 
Nabpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Nabpn/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12X
*abpn/quant_conv2d_6/BiasAdd/ReadVariableOp*abpn/quant_conv2d_6/BiasAdd/ReadVariableOp2ª
Sabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpSabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2®
Uabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Uabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12®
Uabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Uabpn/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Labpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpLabpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2 
Nabpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Nabpn/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12
Labpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpLabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2 
Nabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Nabpn/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:j f
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ô
^
B__inference_lambda_1_layer_call_and_return_conditional_losses_4099

inputs
identity
DepthToSpaceDepthToSpaceinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*

block_sizew
IdentityIdentityDepthToSpace:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
^
B__inference_lambda_2_layer_call_and_return_conditional_losses_7267

inputs
identity\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
clip_by_value/MinimumMinimuminputs clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿs
IdentityIdentityclip_by_value:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï	
þ
-__inference_quant_conv2d_1_layer_call_fn_6541

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_4804
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø#
û
H__inference_quantize_layer_layer_call_and_return_conditional_losses_4936

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity¢#AllValuesQuantize/AssignMaxAllValue¢#AllValuesQuantize/AssignMinAllValue¢8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢(AllValuesQuantize/Maximum/ReadVariableOp¢(AllValuesQuantize/Minimum/ReadVariableOpp
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: r
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: ï
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ï
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(È
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0Ê
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ$
¥
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_4018

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ$
¥
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_3878

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ$
¥
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_3983

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
^
B__inference_lambda_2_layer_call_and_return_conditional_losses_4109

inputs
identity\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
clip_by_value/MinimumMinimuminputs clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿs
IdentityIdentityclip_by_value:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
,
¶
C__inference_quant_add_layer_call_and_return_conditional_losses_4283

inputs
inputs_1@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1j
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             m
MovingAvgQuantize/BatchMinMinadd:z:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             o
MovingAvgQuantize/BatchMaxMaxadd:z:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsadd:z:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
S
	
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_6923

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó	
þ
-__inference_quant_conv2d_5_layer_call_fn_6940

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_4018
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó	
þ
-__inference_quant_conv2d_6_layer_call_fn_7078

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_4063
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°Û
ç#
 __inference__traced_restore_7630
file_prefix<
2assignvariableop_quantize_layer_quantize_layer_min: >
4assignvariableop_1_quantize_layer_quantize_layer_max: :
0assignvariableop_2_quantize_layer_optimizer_step: 8
.assignvariableop_3_quant_conv2d_optimizer_step: 8
*assignvariableop_4_quant_conv2d_kernel_min:8
*assignvariableop_5_quant_conv2d_kernel_max:=
3assignvariableop_6_quant_conv2d_post_activation_min: =
3assignvariableop_7_quant_conv2d_post_activation_max: :
0assignvariableop_8_quant_conv2d_1_optimizer_step: :
,assignvariableop_9_quant_conv2d_1_kernel_min:;
-assignvariableop_10_quant_conv2d_1_kernel_max:@
6assignvariableop_11_quant_conv2d_1_post_activation_min: @
6assignvariableop_12_quant_conv2d_1_post_activation_max: ;
1assignvariableop_13_quant_conv2d_2_optimizer_step: ;
-assignvariableop_14_quant_conv2d_2_kernel_min:;
-assignvariableop_15_quant_conv2d_2_kernel_max:@
6assignvariableop_16_quant_conv2d_2_post_activation_min: @
6assignvariableop_17_quant_conv2d_2_post_activation_max: ;
1assignvariableop_18_quant_conv2d_3_optimizer_step: ;
-assignvariableop_19_quant_conv2d_3_kernel_min:;
-assignvariableop_20_quant_conv2d_3_kernel_max:@
6assignvariableop_21_quant_conv2d_3_post_activation_min: @
6assignvariableop_22_quant_conv2d_3_post_activation_max: ;
1assignvariableop_23_quant_conv2d_4_optimizer_step: ;
-assignvariableop_24_quant_conv2d_4_kernel_min:;
-assignvariableop_25_quant_conv2d_4_kernel_max:@
6assignvariableop_26_quant_conv2d_4_post_activation_min: @
6assignvariableop_27_quant_conv2d_4_post_activation_max: ;
1assignvariableop_28_quant_conv2d_5_optimizer_step: ;
-assignvariableop_29_quant_conv2d_5_kernel_min:;
-assignvariableop_30_quant_conv2d_5_kernel_max:@
6assignvariableop_31_quant_conv2d_5_post_activation_min: @
6assignvariableop_32_quant_conv2d_5_post_activation_max: ;
1assignvariableop_33_quant_conv2d_6_optimizer_step: ;
-assignvariableop_34_quant_conv2d_6_kernel_min:;
-assignvariableop_35_quant_conv2d_6_kernel_max:@
6assignvariableop_36_quant_conv2d_6_post_activation_min: @
6assignvariableop_37_quant_conv2d_6_post_activation_max: 6
,assignvariableop_38_quant_add_optimizer_step: 2
(assignvariableop_39_quant_add_output_min: 2
(assignvariableop_40_quant_add_output_max: ;
!assignvariableop_41_conv2d_kernel:-
assignvariableop_42_conv2d_bias:=
#assignvariableop_43_conv2d_1_kernel:/
!assignvariableop_44_conv2d_1_bias:=
#assignvariableop_45_conv2d_2_kernel:/
!assignvariableop_46_conv2d_2_bias:=
#assignvariableop_47_conv2d_3_kernel:/
!assignvariableop_48_conv2d_3_bias:=
#assignvariableop_49_conv2d_4_kernel:/
!assignvariableop_50_conv2d_4_bias:=
#assignvariableop_51_conv2d_5_kernel:/
!assignvariableop_52_conv2d_5_bias:=
#assignvariableop_53_conv2d_6_kernel:/
!assignvariableop_54_conv2d_6_bias:
identity_56¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*½
value³B°8BBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/output_max/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHá
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¹
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ö
_output_shapesã
à::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp2assignvariableop_quantize_layer_quantize_layer_minIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_1AssignVariableOp4assignvariableop_1_quantize_layer_quantize_layer_maxIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp0assignvariableop_2_quantize_layer_optimizer_stepIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_quant_conv2d_optimizer_stepIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp*assignvariableop_4_quant_conv2d_kernel_minIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp*assignvariableop_5_quant_conv2d_kernel_maxIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_6AssignVariableOp3assignvariableop_6_quant_conv2d_post_activation_minIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_7AssignVariableOp3assignvariableop_7_quant_conv2d_post_activation_maxIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp0assignvariableop_8_quant_conv2d_1_optimizer_stepIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp,assignvariableop_9_quant_conv2d_1_kernel_minIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp-assignvariableop_10_quant_conv2d_1_kernel_maxIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_11AssignVariableOp6assignvariableop_11_quant_conv2d_1_post_activation_minIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_12AssignVariableOp6assignvariableop_12_quant_conv2d_1_post_activation_maxIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_13AssignVariableOp1assignvariableop_13_quant_conv2d_2_optimizer_stepIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp-assignvariableop_14_quant_conv2d_2_kernel_minIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp-assignvariableop_15_quant_conv2d_2_kernel_maxIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_16AssignVariableOp6assignvariableop_16_quant_conv2d_2_post_activation_minIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_17AssignVariableOp6assignvariableop_17_quant_conv2d_2_post_activation_maxIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_18AssignVariableOp1assignvariableop_18_quant_conv2d_3_optimizer_stepIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp-assignvariableop_19_quant_conv2d_3_kernel_minIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp-assignvariableop_20_quant_conv2d_3_kernel_maxIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_21AssignVariableOp6assignvariableop_21_quant_conv2d_3_post_activation_minIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_22AssignVariableOp6assignvariableop_22_quant_conv2d_3_post_activation_maxIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_23AssignVariableOp1assignvariableop_23_quant_conv2d_4_optimizer_stepIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp-assignvariableop_24_quant_conv2d_4_kernel_minIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp-assignvariableop_25_quant_conv2d_4_kernel_maxIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_26AssignVariableOp6assignvariableop_26_quant_conv2d_4_post_activation_minIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_27AssignVariableOp6assignvariableop_27_quant_conv2d_4_post_activation_maxIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_28AssignVariableOp1assignvariableop_28_quant_conv2d_5_optimizer_stepIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp-assignvariableop_29_quant_conv2d_5_kernel_minIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp-assignvariableop_30_quant_conv2d_5_kernel_maxIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_31AssignVariableOp6assignvariableop_31_quant_conv2d_5_post_activation_minIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_32AssignVariableOp6assignvariableop_32_quant_conv2d_5_post_activation_maxIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_33AssignVariableOp1assignvariableop_33_quant_conv2d_6_optimizer_stepIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp-assignvariableop_34_quant_conv2d_6_kernel_minIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp-assignvariableop_35_quant_conv2d_6_kernel_maxIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_36AssignVariableOp6assignvariableop_36_quant_conv2d_6_post_activation_minIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_37AssignVariableOp6assignvariableop_37_quant_conv2d_6_post_activation_maxIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp,assignvariableop_38_quant_add_optimizer_stepIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp(assignvariableop_39_quant_add_output_minIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp(assignvariableop_40_quant_add_output_maxIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp!assignvariableop_41_conv2d_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOpassignvariableop_42_conv2d_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp#assignvariableop_43_conv2d_1_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp!assignvariableop_44_conv2d_1_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp#assignvariableop_45_conv2d_2_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp!assignvariableop_46_conv2d_2_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp#assignvariableop_47_conv2d_3_kernelIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp!assignvariableop_48_conv2d_3_biasIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp#assignvariableop_49_conv2d_4_kernelIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp!assignvariableop_50_conv2d_4_biasIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp#assignvariableop_51_conv2d_5_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp!assignvariableop_52_conv2d_5_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp#assignvariableop_53_conv2d_6_kernelIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp!assignvariableop_54_conv2d_6_biasIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_56IdentityIdentity_55:output:0^NoOp_1*
T0*
_output_shapes
: ö	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_56Identity_56:output:0*
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ó	
þ
-__inference_quant_conv2d_3_layer_call_fn_6732

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_3948
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
ï<
>__inference_abpn_layer_call_and_return_conditional_losses_5991

inputsZ
Pquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: o
Uquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:e
Wquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:e
Wquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource::
,quant_conv2d_biasadd_readvariableop_resource:X
Nquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: Z
Pquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_1_biasadd_readvariableop_resource:Z
Pquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_2_biasadd_readvariableop_resource:Z
Pquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_3_biasadd_readvariableop_resource:Z
Pquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_4_biasadd_readvariableop_resource:Z
Pquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_5_biasadd_readvariableop_resource:Z
Pquant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_6_biasadd_readvariableop_resource:Z
Pquant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: U
Kquant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: W
Mquant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢#quant_conv2d/BiasAdd/ReadVariableOp¢Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_1/BiasAdd/ReadVariableOp¢Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_2/BiasAdd/ReadVariableOp¢Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_3/BiasAdd/ReadVariableOp¢Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_4/BiasAdd/ReadVariableOp¢Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_5/BiasAdd/ReadVariableOp¢Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_6/BiasAdd/ReadVariableOp¢Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Â
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpUquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0â
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpWquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0â
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpWquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0¤
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelTquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(
quant_conv2d/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Gquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#quant_conv2d/BiasAdd/ReadVariableOpReadVariableOp,quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
quant_conv2d/BiasAddBiasAddquant_conv2d/Conv2D:output:0+quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d/ReluReluquant_conv2d/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÌ
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpNquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ð
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpPquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Õ
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d/Relu:activations:0Mquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Oquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0æ
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0æ
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0¬
?quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range( 
quant_conv2d_1/Conv2DConv2D@quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_1/BiasAddBiasAddquant_conv2d_1/Conv2D:output:0-quant_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d_1/ReluReluquant_conv2d_1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ý
8quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_1/Relu:activations:0Oquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0æ
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0æ
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0¬
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(¢
quant_conv2d_2/Conv2DConv2DBquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_2/BiasAddBiasAddquant_conv2d_2/Conv2D:output:0-quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d_2/ReluReluquant_conv2d_2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ý
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_2/Relu:activations:0Oquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0æ
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0æ
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0¬
?quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(¢
quant_conv2d_3/Conv2DConv2DBquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_3/BiasAddBiasAddquant_conv2d_3/Conv2D:output:0-quant_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d_3/ReluReluquant_conv2d_3/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ý
8quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_3/Relu:activations:0Oquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0æ
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0æ
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0¬
?quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(¢
quant_conv2d_4/Conv2DConv2DBquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_4/BiasAddBiasAddquant_conv2d_4/Conv2D:output:0-quant_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d_4/ReluReluquant_conv2d_4/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ý
8quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_4/Relu:activations:0Oquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0æ
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0æ
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0¬
?quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(¢
quant_conv2d_5/Conv2DConv2DBquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_5/BiasAddBiasAddquant_conv2d_5/Conv2D:output:0-quant_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d_5/ReluReluquant_conv2d_5/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Ý
8quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_5/Relu:activations:0Oquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿT
lambda/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
lambda/concatConcatV2Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0lambda/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0æ
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0æ
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0¬
?quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(¢
quant_conv2d_6/Conv2DConv2DBquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_6/BiasAddBiasAddquant_conv2d_6/Conv2D:output:0-quant_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ô
Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0Û
8quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d_6/BiasAdd:output:0Oquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
quant_add/addAddV2lambda/concat:output:0Bquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpKquant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0Ê
Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpMquant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¾
3quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_add/add:z:0Jquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Lquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÂ
lambda_1/DepthToSpaceDepthToSpace=quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*

block_sizee
 lambda_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CÀ
lambda_2/clip_by_value/MinimumMinimumlambda_1/DepthToSpace:output:0)lambda_2/clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
lambda_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
lambda_2/clip_by_valueMaximum"lambda_2/clip_by_value/Minimum:z:0!lambda_2/clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentitylambda_2/clip_by_value:z:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÏ
NoOpNoOpC^quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpE^quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1$^quant_conv2d/BiasAdd/ReadVariableOpM^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpO^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1O^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2F^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpH^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_1/BiasAdd/ReadVariableOpO^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_2/BiasAdd/ReadVariableOpO^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_3/BiasAdd/ReadVariableOpO^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_4/BiasAdd/ReadVariableOpO^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_5/BiasAdd/ReadVariableOpO^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_6/BiasAdd/ReadVariableOpO^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1H^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpBquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12J
#quant_conv2d/BiasAdd/ReadVariableOp#quant_conv2d/BiasAdd/ReadVariableOp2
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpLquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpEquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_1/BiasAdd/ReadVariableOp%quant_conv2d_1/BiasAdd/ReadVariableOp2 
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_2/BiasAdd/ReadVariableOp%quant_conv2d_2/BiasAdd/ReadVariableOp2 
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_3/BiasAdd/ReadVariableOp%quant_conv2d_3/BiasAdd/ReadVariableOp2 
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_4/BiasAdd/ReadVariableOp%quant_conv2d_4/BiasAdd/ReadVariableOp2 
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_5/BiasAdd/ReadVariableOp%quant_conv2d_5/BiasAdd/ReadVariableOp2 
Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_6/BiasAdd/ReadVariableOp%quant_conv2d_6/BiasAdd/ReadVariableOp2 
Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï	
ü
+__inference_quant_conv2d_layer_call_fn_6420

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_3843
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
S
	
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_6611

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
,
¸
C__inference_quant_add_layer_call_and_return_conditional_losses_7221
inputs_0
inputs_1@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1l
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             m
MovingAvgQuantize/BatchMinMinadd:z:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             o
MovingAvgQuantize/BatchMaxMaxadd:z:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsadd:z:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
S
	
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_4888

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾	

@__inference_lambda_layer_call_and_return_conditional_losses_7061
inputs_0
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¥
concatConcatV2inputs_0inputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*É
_input_shapes·
´:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3
Ï$
£
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_6458

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ$
¥
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_6770

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
S
	
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_4636

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ$
¥
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_6562

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
S
	
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_6715

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
û
H__inference_quantize_layer_layer_call_and_return_conditional_losses_3816

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1²
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä

C__inference_quant_add_layer_call_and_return_conditional_losses_7194
inputs_0
inputs_1K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1l
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsadd:z:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ï	
þ
-__inference_quant_conv2d_4_layer_call_fn_6853

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_4552
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
S
	
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_4804

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó	
þ
-__inference_quant_conv2d_1_layer_call_fn_6524

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_3878
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï	
þ
-__inference_quant_conv2d_6_layer_call_fn_7095

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_4358
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ$
¥
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_6666

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö	

(__inference_quant_add_layer_call_fn_7173
inputs_0
inputs_1
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_quant_add_layer_call_and_return_conditional_losses_4088
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ñ$
¥
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_3913

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð


#__inference_abpn_layer_call_fn_5754

inputs
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: #
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: $

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: $

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: $

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: $

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35: 

unknown_36: $

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41: 

unknown_42: 

unknown_43: 

unknown_44: 
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_abpn_layer_call_and_return_conditional_losses_4112
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
S
	
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_4720

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
ÖR
>__inference_abpn_layer_call_and_return_conditional_losses_6355

inputsJ
@quantize_layer_allvaluesquantize_minimum_readvariableop_resource: J
@quantize_layer_allvaluesquantize_maximum_readvariableop_resource: V
<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource:@
2quant_conv2d_lastvaluequant_assignminlast_resource:@
2quant_conv2d_lastvaluequant_assignmaxlast_resource::
,quant_conv2d_biasadd_readvariableop_resource:M
Cquant_conv2d_movingavgquantize_assignminema_readvariableop_resource: M
Cquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_1_lastvaluequant_assignminlast_resource:B
4quant_conv2d_1_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_1_biasadd_readvariableop_resource:O
Equant_conv2d_1_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_2_lastvaluequant_assignminlast_resource:B
4quant_conv2d_2_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_2_biasadd_readvariableop_resource:O
Equant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_3_lastvaluequant_assignminlast_resource:B
4quant_conv2d_3_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_3_biasadd_readvariableop_resource:O
Equant_conv2d_3_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_4_lastvaluequant_assignminlast_resource:B
4quant_conv2d_4_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_4_biasadd_readvariableop_resource:O
Equant_conv2d_4_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_5_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_5_lastvaluequant_assignminlast_resource:B
4quant_conv2d_5_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_5_biasadd_readvariableop_resource:O
Equant_conv2d_5_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_5_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_6_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_6_lastvaluequant_assignminlast_resource:B
4quant_conv2d_6_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_6_biasadd_readvariableop_resource:O
Equant_conv2d_6_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_6_movingavgquantize_assignmaxema_readvariableop_resource: J
@quant_add_movingavgquantize_assignminema_readvariableop_resource: J
@quant_add_movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢<quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢7quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢<quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢7quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢#quant_conv2d/BiasAdd/ReadVariableOp¢)quant_conv2d/LastValueQuant/AssignMaxLast¢)quant_conv2d/LastValueQuant/AssignMinLast¢3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp¢3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp¢Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_1/BiasAdd/ReadVariableOp¢+quant_conv2d_1/LastValueQuant/AssignMaxLast¢+quant_conv2d_1/LastValueQuant/AssignMinLast¢5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp¢5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOp¢Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Aquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢Aquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_2/BiasAdd/ReadVariableOp¢+quant_conv2d_2/LastValueQuant/AssignMaxLast¢+quant_conv2d_2/LastValueQuant/AssignMinLast¢5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp¢5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp¢Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_3/BiasAdd/ReadVariableOp¢+quant_conv2d_3/LastValueQuant/AssignMaxLast¢+quant_conv2d_3/LastValueQuant/AssignMinLast¢5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp¢5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOp¢Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Aquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢Aquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_4/BiasAdd/ReadVariableOp¢+quant_conv2d_4/LastValueQuant/AssignMaxLast¢+quant_conv2d_4/LastValueQuant/AssignMinLast¢5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp¢5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOp¢Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Aquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢Aquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_5/BiasAdd/ReadVariableOp¢+quant_conv2d_5/LastValueQuant/AssignMaxLast¢+quant_conv2d_5/LastValueQuant/AssignMinLast¢5quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOp¢5quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOp¢Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Aquant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢<quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢Aquant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢<quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢%quant_conv2d_6/BiasAdd/ReadVariableOp¢+quant_conv2d_6/LastValueQuant/AssignMaxLast¢+quant_conv2d_6/LastValueQuant/AssignMinLast¢5quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOp¢5quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOp¢Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢Aquant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢<quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢Aquant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢<quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp¢Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢2quantize_layer/AllValuesQuantize/AssignMaxAllValue¢2quantize_layer/AllValuesQuantize/AssignMinAllValue¢Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¢7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp¢7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp
&quantize_layer/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
)quantize_layer/AllValuesQuantize/BatchMinMininputs/quantize_layer/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 
(quantize_layer/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
)quantize_layer/AllValuesQuantize/BatchMaxMaxinputs1quantize_layer/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: °
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0É
(quantize_layer/AllValuesQuantize/MinimumMinimum?quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: q
,quantize_layer/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    »
*quantize_layer/AllValuesQuantize/Minimum_1Minimum,quantize_layer/AllValuesQuantize/Minimum:z:05quantize_layer/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: °
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0É
(quantize_layer/AllValuesQuantize/MaximumMaximum?quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: q
,quantize_layer/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    »
*quantize_layer/AllValuesQuantize/Maximum_1Maximum,quantize_layer/AllValuesQuantize/Maximum:z:05quantize_layer/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: «
2quantize_layer/AllValuesQuantize/AssignMinAllValueAssignVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource.quantize_layer/AllValuesQuantize/Minimum_1:z:08^quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(«
2quantize_layer/AllValuesQuantize/AssignMaxAllValueAssignVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource.quantize_layer/AllValuesQuantize/Maximum_1:z:08^quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(õ
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0÷
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0Â
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
6quant_conv2d/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Î
$quant_conv2d/LastValueQuant/BatchMinMin;quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp:value:0?quant_conv2d/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:¸
3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
6quant_conv2d/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Î
$quant_conv2d/LastValueQuant/BatchMaxMax;quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp:value:0?quant_conv2d/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:j
%quant_conv2d/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿²
#quant_conv2d/LastValueQuant/truedivRealDiv-quant_conv2d/LastValueQuant/BatchMax:output:0.quant_conv2d/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:«
#quant_conv2d/LastValueQuant/MinimumMinimum-quant_conv2d/LastValueQuant/BatchMin:output:0'quant_conv2d/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:f
!quant_conv2d/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¦
quant_conv2d/LastValueQuant/mulMul-quant_conv2d/LastValueQuant/BatchMin:output:0*quant_conv2d/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:§
#quant_conv2d/LastValueQuant/MaximumMaximum-quant_conv2d/LastValueQuant/BatchMax:output:0#quant_conv2d/LastValueQuant/mul:z:0*
T0*
_output_shapes
:Ó
)quant_conv2d/LastValueQuant/AssignMinLastAssignVariableOp2quant_conv2d_lastvaluequant_assignminlast_resource'quant_conv2d/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ó
)quant_conv2d/LastValueQuant/AssignMaxLastAssignVariableOp2quant_conv2d_lastvaluequant_assignmaxlast_resource'quant_conv2d/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ñ
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0é
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp2quant_conv2d_lastvaluequant_assignminlast_resource*^quant_conv2d/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0é
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp2quant_conv2d_lastvaluequant_assignmaxlast_resource*^quant_conv2d/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0¤
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelTquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(
quant_conv2d/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Gquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#quant_conv2d/BiasAdd/ReadVariableOpReadVariableOp,quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
quant_conv2d/BiasAddBiasAddquant_conv2d/Conv2D:output:0+quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d/ReluReluquant_conv2d/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ}
$quant_conv2d/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
'quant_conv2d/MovingAvgQuantize/BatchMinMinquant_conv2d/Relu:activations:0-quant_conv2d/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
&quant_conv2d/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ¡
'quant_conv2d/MovingAvgQuantize/BatchMaxMaxquant_conv2d/Relu:activations:0/quant_conv2d/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: m
(quant_conv2d/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ·
&quant_conv2d/MovingAvgQuantize/MinimumMinimum0quant_conv2d/MovingAvgQuantize/BatchMin:output:01quant_conv2d/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: m
(quant_conv2d/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ·
&quant_conv2d/MovingAvgQuantize/MaximumMaximum0quant_conv2d/MovingAvgQuantize/BatchMax:output:01quant_conv2d/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: v
1quant_conv2d/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Ç
/quant_conv2d/MovingAvgQuantize/AssignMinEma/subSubBquant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0*quant_conv2d/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: È
/quant_conv2d/MovingAvgQuantize/AssignMinEma/mulMul3quant_conv2d/MovingAvgQuantize/AssignMinEma/sub:z:0:quant_conv2d/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: °
?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource3quant_conv2d/MovingAvgQuantize/AssignMinEma/mul:z:0;^quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0v
1quant_conv2d/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Ç
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/subSubBquant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0*quant_conv2d/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: È
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/mulMul3quant_conv2d/MovingAvgQuantize/AssignMaxEma/sub:z:0:quant_conv2d/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: °
?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource3quant_conv2d/MovingAvgQuantize/AssignMaxEma/mul:z:0;^quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource@^quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource@^quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Õ
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d/Relu:activations:0Mquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Oquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_1/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_1/LastValueQuant/BatchMinMin=quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_1/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:¼
5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_1/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_1/LastValueQuant/BatchMaxMax=quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_1/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:l
'quant_conv2d_1/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¸
%quant_conv2d_1/LastValueQuant/truedivRealDiv/quant_conv2d_1/LastValueQuant/BatchMax:output:00quant_conv2d_1/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:±
%quant_conv2d_1/LastValueQuant/MinimumMinimum/quant_conv2d_1/LastValueQuant/BatchMin:output:0)quant_conv2d_1/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:h
#quant_conv2d_1/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¬
!quant_conv2d_1/LastValueQuant/mulMul/quant_conv2d_1/LastValueQuant/BatchMin:output:0,quant_conv2d_1/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:­
%quant_conv2d_1/LastValueQuant/MaximumMaximum/quant_conv2d_1/LastValueQuant/BatchMax:output:0%quant_conv2d_1/LastValueQuant/mul:z:0*
T0*
_output_shapes
:Ù
+quant_conv2d_1/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_1_lastvaluequant_assignminlast_resource)quant_conv2d_1/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ù
+quant_conv2d_1/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_1_lastvaluequant_assignmaxlast_resource)quant_conv2d_1/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Õ
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0ï
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_1_lastvaluequant_assignminlast_resource,^quant_conv2d_1/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0ï
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_1_lastvaluequant_assignmaxlast_resource,^quant_conv2d_1/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0¬
?quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range( 
quant_conv2d_1/Conv2DConv2D@quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_1/BiasAddBiasAddquant_conv2d_1/Conv2D:output:0-quant_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d_1/ReluReluquant_conv2d_1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&quant_conv2d_1/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¥
)quant_conv2d_1/MovingAvgQuantize/BatchMinMin!quant_conv2d_1/Relu:activations:0/quant_conv2d_1/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
(quant_conv2d_1/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             §
)quant_conv2d_1/MovingAvgQuantize/BatchMaxMax!quant_conv2d_1/Relu:activations:01quant_conv2d_1/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_conv2d_1/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_1/MovingAvgQuantize/MinimumMinimum2quant_conv2d_1/MovingAvgQuantize/BatchMin:output:03quant_conv2d_1/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_conv2d_1/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_1/MovingAvgQuantize/MaximumMaximum2quant_conv2d_1/MovingAvgQuantize/BatchMax:output:03quant_conv2d_1/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_conv2d_1/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_1_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_1/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_1/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_1/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_1/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_1_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_1/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_1/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_1_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Ý
8quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_1/Relu:activations:0Oquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_2/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_2/LastValueQuant/BatchMinMin=quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_2/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:¼
5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_2/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_2/LastValueQuant/BatchMaxMax=quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_2/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:l
'quant_conv2d_2/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¸
%quant_conv2d_2/LastValueQuant/truedivRealDiv/quant_conv2d_2/LastValueQuant/BatchMax:output:00quant_conv2d_2/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:±
%quant_conv2d_2/LastValueQuant/MinimumMinimum/quant_conv2d_2/LastValueQuant/BatchMin:output:0)quant_conv2d_2/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:h
#quant_conv2d_2/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¬
!quant_conv2d_2/LastValueQuant/mulMul/quant_conv2d_2/LastValueQuant/BatchMin:output:0,quant_conv2d_2/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:­
%quant_conv2d_2/LastValueQuant/MaximumMaximum/quant_conv2d_2/LastValueQuant/BatchMax:output:0%quant_conv2d_2/LastValueQuant/mul:z:0*
T0*
_output_shapes
:Ù
+quant_conv2d_2/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_2_lastvaluequant_assignminlast_resource)quant_conv2d_2/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ù
+quant_conv2d_2/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_2_lastvaluequant_assignmaxlast_resource)quant_conv2d_2/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Õ
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0ï
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_2_lastvaluequant_assignminlast_resource,^quant_conv2d_2/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0ï
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_2_lastvaluequant_assignmaxlast_resource,^quant_conv2d_2/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0¬
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(¢
quant_conv2d_2/Conv2DConv2DBquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_2/BiasAddBiasAddquant_conv2d_2/Conv2D:output:0-quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d_2/ReluReluquant_conv2d_2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&quant_conv2d_2/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¥
)quant_conv2d_2/MovingAvgQuantize/BatchMinMin!quant_conv2d_2/Relu:activations:0/quant_conv2d_2/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
(quant_conv2d_2/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             §
)quant_conv2d_2/MovingAvgQuantize/BatchMaxMax!quant_conv2d_2/Relu:activations:01quant_conv2d_2/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_conv2d_2/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_2/MovingAvgQuantize/MinimumMinimum2quant_conv2d_2/MovingAvgQuantize/BatchMin:output:03quant_conv2d_2/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_conv2d_2/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_2/MovingAvgQuantize/MaximumMaximum2quant_conv2d_2/MovingAvgQuantize/BatchMax:output:03quant_conv2d_2/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_conv2d_2/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_2/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_2/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_2/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_2/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Ý
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_2/Relu:activations:0Oquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_3/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_3/LastValueQuant/BatchMinMin=quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_3/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:¼
5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_3/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_3/LastValueQuant/BatchMaxMax=quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_3/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:l
'quant_conv2d_3/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¸
%quant_conv2d_3/LastValueQuant/truedivRealDiv/quant_conv2d_3/LastValueQuant/BatchMax:output:00quant_conv2d_3/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:±
%quant_conv2d_3/LastValueQuant/MinimumMinimum/quant_conv2d_3/LastValueQuant/BatchMin:output:0)quant_conv2d_3/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:h
#quant_conv2d_3/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¬
!quant_conv2d_3/LastValueQuant/mulMul/quant_conv2d_3/LastValueQuant/BatchMin:output:0,quant_conv2d_3/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:­
%quant_conv2d_3/LastValueQuant/MaximumMaximum/quant_conv2d_3/LastValueQuant/BatchMax:output:0%quant_conv2d_3/LastValueQuant/mul:z:0*
T0*
_output_shapes
:Ù
+quant_conv2d_3/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_3_lastvaluequant_assignminlast_resource)quant_conv2d_3/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ù
+quant_conv2d_3/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_3_lastvaluequant_assignmaxlast_resource)quant_conv2d_3/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Õ
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0ï
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_3_lastvaluequant_assignminlast_resource,^quant_conv2d_3/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0ï
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_3_lastvaluequant_assignmaxlast_resource,^quant_conv2d_3/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0¬
?quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(¢
quant_conv2d_3/Conv2DConv2DBquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_3/BiasAddBiasAddquant_conv2d_3/Conv2D:output:0-quant_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d_3/ReluReluquant_conv2d_3/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&quant_conv2d_3/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¥
)quant_conv2d_3/MovingAvgQuantize/BatchMinMin!quant_conv2d_3/Relu:activations:0/quant_conv2d_3/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
(quant_conv2d_3/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             §
)quant_conv2d_3/MovingAvgQuantize/BatchMaxMax!quant_conv2d_3/Relu:activations:01quant_conv2d_3/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_conv2d_3/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_3/MovingAvgQuantize/MinimumMinimum2quant_conv2d_3/MovingAvgQuantize/BatchMin:output:03quant_conv2d_3/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_conv2d_3/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_3/MovingAvgQuantize/MaximumMaximum2quant_conv2d_3/MovingAvgQuantize/BatchMax:output:03quant_conv2d_3/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_conv2d_3/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_3_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_3/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_3/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_3/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_3/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_3_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_3/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_3/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_3_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Ý
8quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_3/Relu:activations:0Oquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_4/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_4/LastValueQuant/BatchMinMin=quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_4/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:¼
5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_4/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_4/LastValueQuant/BatchMaxMax=quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_4/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:l
'quant_conv2d_4/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¸
%quant_conv2d_4/LastValueQuant/truedivRealDiv/quant_conv2d_4/LastValueQuant/BatchMax:output:00quant_conv2d_4/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:±
%quant_conv2d_4/LastValueQuant/MinimumMinimum/quant_conv2d_4/LastValueQuant/BatchMin:output:0)quant_conv2d_4/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:h
#quant_conv2d_4/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¬
!quant_conv2d_4/LastValueQuant/mulMul/quant_conv2d_4/LastValueQuant/BatchMin:output:0,quant_conv2d_4/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:­
%quant_conv2d_4/LastValueQuant/MaximumMaximum/quant_conv2d_4/LastValueQuant/BatchMax:output:0%quant_conv2d_4/LastValueQuant/mul:z:0*
T0*
_output_shapes
:Ù
+quant_conv2d_4/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_4_lastvaluequant_assignminlast_resource)quant_conv2d_4/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ù
+quant_conv2d_4/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_4_lastvaluequant_assignmaxlast_resource)quant_conv2d_4/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Õ
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0ï
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_4_lastvaluequant_assignminlast_resource,^quant_conv2d_4/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0ï
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_4_lastvaluequant_assignmaxlast_resource,^quant_conv2d_4/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0¬
?quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(¢
quant_conv2d_4/Conv2DConv2DBquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_4/BiasAddBiasAddquant_conv2d_4/Conv2D:output:0-quant_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d_4/ReluReluquant_conv2d_4/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&quant_conv2d_4/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¥
)quant_conv2d_4/MovingAvgQuantize/BatchMinMin!quant_conv2d_4/Relu:activations:0/quant_conv2d_4/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
(quant_conv2d_4/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             §
)quant_conv2d_4/MovingAvgQuantize/BatchMaxMax!quant_conv2d_4/Relu:activations:01quant_conv2d_4/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_conv2d_4/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_4/MovingAvgQuantize/MinimumMinimum2quant_conv2d_4/MovingAvgQuantize/BatchMin:output:03quant_conv2d_4/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_conv2d_4/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_4/MovingAvgQuantize/MaximumMaximum2quant_conv2d_4/MovingAvgQuantize/BatchMax:output:03quant_conv2d_4/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_conv2d_4/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_4_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_4/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_4/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_4/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_4/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_4_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_4/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_4/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_4_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Ý
8quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_4/Relu:activations:0Oquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
5quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_5_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_5/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_5/LastValueQuant/BatchMinMin=quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_5/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:¼
5quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_5_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_5/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_5/LastValueQuant/BatchMaxMax=quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_5/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:l
'quant_conv2d_5/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¸
%quant_conv2d_5/LastValueQuant/truedivRealDiv/quant_conv2d_5/LastValueQuant/BatchMax:output:00quant_conv2d_5/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:±
%quant_conv2d_5/LastValueQuant/MinimumMinimum/quant_conv2d_5/LastValueQuant/BatchMin:output:0)quant_conv2d_5/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:h
#quant_conv2d_5/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¬
!quant_conv2d_5/LastValueQuant/mulMul/quant_conv2d_5/LastValueQuant/BatchMin:output:0,quant_conv2d_5/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:­
%quant_conv2d_5/LastValueQuant/MaximumMaximum/quant_conv2d_5/LastValueQuant/BatchMax:output:0%quant_conv2d_5/LastValueQuant/mul:z:0*
T0*
_output_shapes
:Ù
+quant_conv2d_5/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_5_lastvaluequant_assignminlast_resource)quant_conv2d_5/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ù
+quant_conv2d_5/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_5_lastvaluequant_assignmaxlast_resource)quant_conv2d_5/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Õ
Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_5_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0ï
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_5_lastvaluequant_assignminlast_resource,^quant_conv2d_5/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0ï
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_5_lastvaluequant_assignmaxlast_resource,^quant_conv2d_5/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0¬
?quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(¢
quant_conv2d_5/Conv2DConv2DBquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_5/BiasAddBiasAddquant_conv2d_5/Conv2D:output:0-quant_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
quant_conv2d_5/ReluReluquant_conv2d_5/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&quant_conv2d_5/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¥
)quant_conv2d_5/MovingAvgQuantize/BatchMinMin!quant_conv2d_5/Relu:activations:0/quant_conv2d_5/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
(quant_conv2d_5/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             §
)quant_conv2d_5/MovingAvgQuantize/BatchMaxMax!quant_conv2d_5/Relu:activations:01quant_conv2d_5/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_conv2d_5/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_5/MovingAvgQuantize/MinimumMinimum2quant_conv2d_5/MovingAvgQuantize/BatchMin:output:03quant_conv2d_5/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_conv2d_5/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_5/MovingAvgQuantize/MaximumMaximum2quant_conv2d_5/MovingAvgQuantize/BatchMax:output:03quant_conv2d_5/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_conv2d_5/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_5_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_5/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_5/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_5/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_5/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_5/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_5_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_5/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_5_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_5/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_5_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_5_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_5_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Ý
8quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_5/Relu:activations:0Oquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿT
lambda/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
lambda/concatConcatV2Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0lambda/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
5quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_6_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_6/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_6/LastValueQuant/BatchMinMin=quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_6/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:¼
5quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_6_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0
8quant_conv2d_6/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
&quant_conv2d_6/LastValueQuant/BatchMaxMax=quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_6/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:l
'quant_conv2d_6/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¸
%quant_conv2d_6/LastValueQuant/truedivRealDiv/quant_conv2d_6/LastValueQuant/BatchMax:output:00quant_conv2d_6/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:±
%quant_conv2d_6/LastValueQuant/MinimumMinimum/quant_conv2d_6/LastValueQuant/BatchMin:output:0)quant_conv2d_6/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:h
#quant_conv2d_6/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿¬
!quant_conv2d_6/LastValueQuant/mulMul/quant_conv2d_6/LastValueQuant/BatchMin:output:0,quant_conv2d_6/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:­
%quant_conv2d_6/LastValueQuant/MaximumMaximum/quant_conv2d_6/LastValueQuant/BatchMax:output:0%quant_conv2d_6/LastValueQuant/mul:z:0*
T0*
_output_shapes
:Ù
+quant_conv2d_6/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_6_lastvaluequant_assignminlast_resource)quant_conv2d_6/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Ù
+quant_conv2d_6/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_6_lastvaluequant_assignmaxlast_resource)quant_conv2d_6/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(Õ
Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_6_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0ï
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_6_lastvaluequant_assignminlast_resource,^quant_conv2d_6/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0ï
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_6_lastvaluequant_assignmaxlast_resource,^quant_conv2d_6/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0¬
?quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(¢
quant_conv2d_6/Conv2DConv2DBquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%quant_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
quant_conv2d_6/BiasAddBiasAddquant_conv2d_6/Conv2D:output:0-quant_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&quant_conv2d_6/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             £
)quant_conv2d_6/MovingAvgQuantize/BatchMinMinquant_conv2d_6/BiasAdd:output:0/quant_conv2d_6/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 
(quant_conv2d_6/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ¥
)quant_conv2d_6/MovingAvgQuantize/BatchMaxMaxquant_conv2d_6/BiasAdd:output:01quant_conv2d_6/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_conv2d_6/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_6/MovingAvgQuantize/MinimumMinimum2quant_conv2d_6/MovingAvgQuantize/BatchMin:output:03quant_conv2d_6/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_conv2d_6/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
(quant_conv2d_6/MovingAvgQuantize/MaximumMaximum2quant_conv2d_6/MovingAvgQuantize/BatchMax:output:03quant_conv2d_6/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_conv2d_6/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_6_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_6/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_6/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_6/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_6/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_6/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_6_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_6/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
<quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_6_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0Í
1quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_6/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: Î
1quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¸
Aquant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_6_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0
Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_6_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_6_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Û
8quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d_6/BiasAdd:output:0Oquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
quant_add/addAddV2lambda/concat:output:0Bquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿz
!quant_add/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
$quant_add/MovingAvgQuantize/BatchMinMinquant_add/add:z:0*quant_add/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: |
#quant_add/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
$quant_add/MovingAvgQuantize/BatchMaxMaxquant_add/add:z:0,quant_add/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: j
%quant_add/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
#quant_add/MovingAvgQuantize/MinimumMinimum-quant_add/MovingAvgQuantize/BatchMin:output:0.quant_add/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: j
%quant_add/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
#quant_add/MovingAvgQuantize/MaximumMaximum-quant_add/MovingAvgQuantize/BatchMax:output:0.quant_add/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: s
.quant_add/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:°
7quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp@quant_add_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0¾
,quant_add/MovingAvgQuantize/AssignMinEma/subSub?quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0'quant_add/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¿
,quant_add/MovingAvgQuantize/AssignMinEma/mulMul0quant_add/MovingAvgQuantize/AssignMinEma/sub:z:07quant_add/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ¤
<quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp@quant_add_movingavgquantize_assignminema_readvariableop_resource0quant_add/MovingAvgQuantize/AssignMinEma/mul:z:08^quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0s
.quant_add/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:°
7quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp@quant_add_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0¾
,quant_add/MovingAvgQuantize/AssignMaxEma/subSub?quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0'quant_add/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¿
,quant_add/MovingAvgQuantize/AssignMaxEma/mulMul0quant_add/MovingAvgQuantize/AssignMaxEma/sub:z:07quant_add/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ¤
<quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp@quant_add_movingavgquantize_assignmaxema_readvariableop_resource0quant_add/MovingAvgQuantize/AssignMaxEma/mul:z:08^quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0ú
Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp@quant_add_movingavgquantize_assignminema_readvariableop_resource=^quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0ü
Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@quant_add_movingavgquantize_assignmaxema_readvariableop_resource=^quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¾
3quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_add/add:z:0Jquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Lquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÂ
lambda_1/DepthToSpaceDepthToSpace=quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*

block_sizee
 lambda_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CÀ
lambda_2/clip_by_value/MinimumMinimumlambda_1/DepthToSpace:output:0)lambda_2/clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
lambda_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
lambda_2/clip_by_valueMaximum"lambda_2/clip_by_value/Minimum:z:0!lambda_2/clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentitylambda_2/clip_by_value:z:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÍ7
NoOpNoOp=^quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp8^quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOp=^quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp8^quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOpC^quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpE^quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1$^quant_conv2d/BiasAdd/ReadVariableOp*^quant_conv2d/LastValueQuant/AssignMaxLast*^quant_conv2d/LastValueQuant/AssignMinLast4^quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp4^quant_conv2d/LastValueQuant/BatchMin/ReadVariableOpM^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpO^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1O^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2@^quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp;^quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp@^quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp;^quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOpF^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpH^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_1/BiasAdd/ReadVariableOp,^quant_conv2d_1/LastValueQuant/AssignMaxLast,^quant_conv2d_1/LastValueQuant/AssignMinLast6^quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_2/BiasAdd/ReadVariableOp,^quant_conv2d_2/LastValueQuant/AssignMaxLast,^quant_conv2d_2/LastValueQuant/AssignMinLast6^quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_3/BiasAdd/ReadVariableOp,^quant_conv2d_3/LastValueQuant/AssignMaxLast,^quant_conv2d_3/LastValueQuant/AssignMinLast6^quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_4/BiasAdd/ReadVariableOp,^quant_conv2d_4/LastValueQuant/AssignMaxLast,^quant_conv2d_4/LastValueQuant/AssignMinLast6^quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_5/BiasAdd/ReadVariableOp,^quant_conv2d_5/LastValueQuant/AssignMaxLast,^quant_conv2d_5/LastValueQuant/AssignMinLast6^quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_6/BiasAdd/ReadVariableOp,^quant_conv2d_6/LastValueQuant/AssignMaxLast,^quant_conv2d_6/LastValueQuant/AssignMinLast6^quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_13^quantize_layer/AllValuesQuantize/AssignMaxAllValue3^quantize_layer/AllValuesQuantize/AssignMinAllValueH^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_18^quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp8^quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2|
<quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp<quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2r
7quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOp7quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2|
<quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp<quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2r
7quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOp7quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpBquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12J
#quant_conv2d/BiasAdd/ReadVariableOp#quant_conv2d/BiasAdd/ReadVariableOp2V
)quant_conv2d/LastValueQuant/AssignMaxLast)quant_conv2d/LastValueQuant/AssignMaxLast2V
)quant_conv2d/LastValueQuant/AssignMinLast)quant_conv2d/LastValueQuant/AssignMinLast2j
3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp2j
3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp2
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpLquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2x
:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2x
:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpEquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_1/BiasAdd/ReadVariableOp%quant_conv2d_1/BiasAdd/ReadVariableOp2Z
+quant_conv2d_1/LastValueQuant/AssignMaxLast+quant_conv2d_1/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_1/LastValueQuant/AssignMinLast+quant_conv2d_1/LastValueQuant/AssignMinLast2n
5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_2/BiasAdd/ReadVariableOp%quant_conv2d_2/BiasAdd/ReadVariableOp2Z
+quant_conv2d_2/LastValueQuant/AssignMaxLast+quant_conv2d_2/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_2/LastValueQuant/AssignMinLast+quant_conv2d_2/LastValueQuant/AssignMinLast2n
5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_3/BiasAdd/ReadVariableOp%quant_conv2d_3/BiasAdd/ReadVariableOp2Z
+quant_conv2d_3/LastValueQuant/AssignMaxLast+quant_conv2d_3/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_3/LastValueQuant/AssignMinLast+quant_conv2d_3/LastValueQuant/AssignMinLast2n
5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_4/BiasAdd/ReadVariableOp%quant_conv2d_4/BiasAdd/ReadVariableOp2Z
+quant_conv2d_4/LastValueQuant/AssignMaxLast+quant_conv2d_4/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_4/LastValueQuant/AssignMinLast+quant_conv2d_4/LastValueQuant/AssignMinLast2n
5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_5/BiasAdd/ReadVariableOp%quant_conv2d_5/BiasAdd/ReadVariableOp2Z
+quant_conv2d_5/LastValueQuant/AssignMaxLast+quant_conv2d_5/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_5/LastValueQuant/AssignMinLast+quant_conv2d_5/LastValueQuant/AssignMinLast2n
5quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_6/BiasAdd/ReadVariableOp%quant_conv2d_6/BiasAdd/ReadVariableOp2Z
+quant_conv2d_6/LastValueQuant/AssignMaxLast+quant_conv2d_6/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_6/LastValueQuant/AssignMinLast+quant_conv2d_6/LastValueQuant/AssignMinLast2n
5quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2¤
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12¤
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12h
2quantize_layer/AllValuesQuantize/AssignMaxAllValue2quantize_layer/AllValuesQuantize/AssignMaxAllValue2h
2quantize_layer/AllValuesQuantize/AssignMinAllValue2quantize_layer/AllValuesQuantize/AssignMinAllValue2
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp2r
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

C
'__inference_lambda_1_layer_call_fn_7231

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_1_layer_call_and_return_conditional_losses_4239z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ$
¥
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_3948

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
S
	
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_4552

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³


#__inference_abpn_layer_call_fn_5342
input_1
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: #
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: $

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: $

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: $

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: $

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35: 

unknown_36: $

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41: 

unknown_42: 

unknown_43: 

unknown_44: 
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	!$'**-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_abpn_layer_call_and_return_conditional_losses_5150
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ï	
þ
-__inference_quant_conv2d_3_layer_call_fn_6749

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_4636
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÈK

>__inference_abpn_layer_call_and_return_conditional_losses_5450
input_1
quantize_layer_5345: 
quantize_layer_5347: +
quant_conv2d_5350:
quant_conv2d_5352:
quant_conv2d_5354:
quant_conv2d_5356:
quant_conv2d_5358: 
quant_conv2d_5360: -
quant_conv2d_1_5363:!
quant_conv2d_1_5365:!
quant_conv2d_1_5367:!
quant_conv2d_1_5369:
quant_conv2d_1_5371: 
quant_conv2d_1_5373: -
quant_conv2d_2_5376:!
quant_conv2d_2_5378:!
quant_conv2d_2_5380:!
quant_conv2d_2_5382:
quant_conv2d_2_5384: 
quant_conv2d_2_5386: -
quant_conv2d_3_5389:!
quant_conv2d_3_5391:!
quant_conv2d_3_5393:!
quant_conv2d_3_5395:
quant_conv2d_3_5397: 
quant_conv2d_3_5399: -
quant_conv2d_4_5402:!
quant_conv2d_4_5404:!
quant_conv2d_4_5406:!
quant_conv2d_4_5408:
quant_conv2d_4_5410: 
quant_conv2d_4_5412: -
quant_conv2d_5_5415:!
quant_conv2d_5_5417:!
quant_conv2d_5_5419:!
quant_conv2d_5_5421:
quant_conv2d_5_5423: 
quant_conv2d_5_5425: -
quant_conv2d_6_5429:!
quant_conv2d_6_5431:!
quant_conv2d_6_5433:!
quant_conv2d_6_5435:
quant_conv2d_6_5437: 
quant_conv2d_6_5439: 
quant_add_5442: 
quant_add_5444: 
identity¢!quant_add/StatefulPartitionedCall¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢&quant_conv2d_3/StatefulPartitionedCall¢&quant_conv2d_4/StatefulPartitionedCall¢&quant_conv2d_5/StatefulPartitionedCall¢&quant_conv2d_6/StatefulPartitionedCall¢&quantize_layer/StatefulPartitionedCall
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1quantize_layer_5345quantize_layer_5347*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantize_layer_layer_call_and_return_conditional_losses_3816
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_5350quant_conv2d_5352quant_conv2d_5354quant_conv2d_5356quant_conv2d_5358quant_conv2d_5360*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_3843
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0quant_conv2d_1_5363quant_conv2d_1_5365quant_conv2d_1_5367quant_conv2d_1_5369quant_conv2d_1_5371quant_conv2d_1_5373*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_3878¡
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0quant_conv2d_2_5376quant_conv2d_2_5378quant_conv2d_2_5380quant_conv2d_2_5382quant_conv2d_2_5384quant_conv2d_2_5386*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_3913¡
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_5389quant_conv2d_3_5391quant_conv2d_3_5393quant_conv2d_3_5395quant_conv2d_3_5397quant_conv2d_3_5399*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_3948¡
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_5402quant_conv2d_4_5404quant_conv2d_4_5406quant_conv2d_4_5408quant_conv2d_4_5410quant_conv2d_4_5412*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_3983¡
&quant_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0quant_conv2d_5_5415quant_conv2d_5_5417quant_conv2d_5_5419quant_conv2d_5_5421quant_conv2d_5_5423quant_conv2d_5_5425*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_4018
lambda/PartitionedCallPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_4041¡
&quant_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_5/StatefulPartitionedCall:output:0quant_conv2d_6_5429quant_conv2d_6_5431quant_conv2d_6_5433quant_conv2d_6_5435quant_conv2d_6_5437quant_conv2d_6_5439*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_4063Ó
!quant_add/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0/quant_conv2d_6/StatefulPartitionedCall:output:0quant_add_5442quant_add_5444*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_quant_add_layer_call_and_return_conditional_losses_4088ô
lambda_1/PartitionedCallPartitionedCall*quant_add/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_1_layer_call_and_return_conditional_losses_4099ë
lambda_2/PartitionedCallPartitionedCall!lambda_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_2_layer_call_and_return_conditional_losses_4109
IdentityIdentity!lambda_2/PartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp"^quant_add/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall'^quant_conv2d_5/StatefulPartitionedCall'^quant_conv2d_6/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_add/StatefulPartitionedCall!quant_add/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2P
&quant_conv2d_5/StatefulPartitionedCall&quant_conv2d_5/StatefulPartitionedCall2P
&quant_conv2d_6/StatefulPartitionedCall&quant_conv2d_6/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:j f
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ÅK

>__inference_abpn_layer_call_and_return_conditional_losses_4112

inputs
quantize_layer_3817: 
quantize_layer_3819: +
quant_conv2d_3844:
quant_conv2d_3846:
quant_conv2d_3848:
quant_conv2d_3850:
quant_conv2d_3852: 
quant_conv2d_3854: -
quant_conv2d_1_3879:!
quant_conv2d_1_3881:!
quant_conv2d_1_3883:!
quant_conv2d_1_3885:
quant_conv2d_1_3887: 
quant_conv2d_1_3889: -
quant_conv2d_2_3914:!
quant_conv2d_2_3916:!
quant_conv2d_2_3918:!
quant_conv2d_2_3920:
quant_conv2d_2_3922: 
quant_conv2d_2_3924: -
quant_conv2d_3_3949:!
quant_conv2d_3_3951:!
quant_conv2d_3_3953:!
quant_conv2d_3_3955:
quant_conv2d_3_3957: 
quant_conv2d_3_3959: -
quant_conv2d_4_3984:!
quant_conv2d_4_3986:!
quant_conv2d_4_3988:!
quant_conv2d_4_3990:
quant_conv2d_4_3992: 
quant_conv2d_4_3994: -
quant_conv2d_5_4019:!
quant_conv2d_5_4021:!
quant_conv2d_5_4023:!
quant_conv2d_5_4025:
quant_conv2d_5_4027: 
quant_conv2d_5_4029: -
quant_conv2d_6_4064:!
quant_conv2d_6_4066:!
quant_conv2d_6_4068:!
quant_conv2d_6_4070:
quant_conv2d_6_4072: 
quant_conv2d_6_4074: 
quant_add_4089: 
quant_add_4091: 
identity¢!quant_add/StatefulPartitionedCall¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢&quant_conv2d_3/StatefulPartitionedCall¢&quant_conv2d_4/StatefulPartitionedCall¢&quant_conv2d_5/StatefulPartitionedCall¢&quant_conv2d_6/StatefulPartitionedCall¢&quantize_layer/StatefulPartitionedCall
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_3817quantize_layer_3819*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantize_layer_layer_call_and_return_conditional_losses_3816
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_3844quant_conv2d_3846quant_conv2d_3848quant_conv2d_3850quant_conv2d_3852quant_conv2d_3854*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_3843
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0quant_conv2d_1_3879quant_conv2d_1_3881quant_conv2d_1_3883quant_conv2d_1_3885quant_conv2d_1_3887quant_conv2d_1_3889*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_3878¡
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0quant_conv2d_2_3914quant_conv2d_2_3916quant_conv2d_2_3918quant_conv2d_2_3920quant_conv2d_2_3922quant_conv2d_2_3924*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_3913¡
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_3949quant_conv2d_3_3951quant_conv2d_3_3953quant_conv2d_3_3955quant_conv2d_3_3957quant_conv2d_3_3959*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_3948¡
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_3984quant_conv2d_4_3986quant_conv2d_4_3988quant_conv2d_4_3990quant_conv2d_4_3992quant_conv2d_4_3994*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_3983¡
&quant_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0quant_conv2d_5_4019quant_conv2d_5_4021quant_conv2d_5_4023quant_conv2d_5_4025quant_conv2d_5_4027quant_conv2d_5_4029*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_4018
lambda/PartitionedCallPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_4041¡
&quant_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_5/StatefulPartitionedCall:output:0quant_conv2d_6_4064quant_conv2d_6_4066quant_conv2d_6_4068quant_conv2d_6_4070quant_conv2d_6_4072quant_conv2d_6_4074*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_4063Ó
!quant_add/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0/quant_conv2d_6/StatefulPartitionedCall:output:0quant_add_4089quant_add_4091*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_quant_add_layer_call_and_return_conditional_losses_4088ô
lambda_1/PartitionedCallPartitionedCall*quant_add/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_1_layer_call_and_return_conditional_losses_4099ë
lambda_2/PartitionedCallPartitionedCall!lambda_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_2_layer_call_and_return_conditional_losses_4109
IdentityIdentity!lambda_2/PartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp"^quant_add/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall'^quant_conv2d_5/StatefulPartitionedCall'^quant_conv2d_6/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_add/StatefulPartitionedCall!quant_add/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2P
&quant_conv2d_5/StatefulPartitionedCall&quant_conv2d_5/StatefulPartitionedCall2P
&quant_conv2d_6/StatefulPartitionedCall&quant_conv2d_6/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

C
'__inference_lambda_2_layer_call_fn_7246

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lambda_2_layer_call_and_return_conditional_losses_4109z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã#
¥
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_4063

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü

C__inference_quant_add_layer_call_and_return_conditional_losses_4088

inputs
inputs_1K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1j
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsadd:z:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦R
	
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_4358

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢LastValueQuant/AssignMaxLast¢LastValueQuant/AssignMinLast¢&LastValueQuant/BatchMax/ReadVariableOp¢&LastValueQuant/BatchMin/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMaxEma/ReadVariableOp¢2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp¢-MovingAvgQuantize/AssignMinEma/ReadVariableOp¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0~
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:¬
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(¬
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(·
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype0Â
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: r
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0 
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: ¡
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: ü
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0Ü
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0Þ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²l
Û
__inference__traced_save_7455
file_prefix@
<savev2_quantize_layer_quantize_layer_min_read_readvariableop@
<savev2_quantize_layer_quantize_layer_max_read_readvariableop<
8savev2_quantize_layer_optimizer_step_read_readvariableop:
6savev2_quant_conv2d_optimizer_step_read_readvariableop6
2savev2_quant_conv2d_kernel_min_read_readvariableop6
2savev2_quant_conv2d_kernel_max_read_readvariableop?
;savev2_quant_conv2d_post_activation_min_read_readvariableop?
;savev2_quant_conv2d_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_1_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_1_kernel_min_read_readvariableop8
4savev2_quant_conv2d_1_kernel_max_read_readvariableopA
=savev2_quant_conv2d_1_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_1_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_2_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_2_kernel_min_read_readvariableop8
4savev2_quant_conv2d_2_kernel_max_read_readvariableopA
=savev2_quant_conv2d_2_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_2_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_3_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_3_kernel_min_read_readvariableop8
4savev2_quant_conv2d_3_kernel_max_read_readvariableopA
=savev2_quant_conv2d_3_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_3_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_4_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_4_kernel_min_read_readvariableop8
4savev2_quant_conv2d_4_kernel_max_read_readvariableopA
=savev2_quant_conv2d_4_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_4_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_5_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_5_kernel_min_read_readvariableop8
4savev2_quant_conv2d_5_kernel_max_read_readvariableopA
=savev2_quant_conv2d_5_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_5_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_6_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_6_kernel_min_read_readvariableop8
4savev2_quant_conv2d_6_kernel_max_read_readvariableopA
=savev2_quant_conv2d_6_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_6_post_activation_max_read_readvariableop7
3savev2_quant_add_optimizer_step_read_readvariableop3
/savev2_quant_add_output_min_read_readvariableop3
/savev2_quant_add_output_max_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*½
value³B°8BBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/output_max/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÞ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ü
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_quantize_layer_quantize_layer_min_read_readvariableop<savev2_quantize_layer_quantize_layer_max_read_readvariableop8savev2_quantize_layer_optimizer_step_read_readvariableop6savev2_quant_conv2d_optimizer_step_read_readvariableop2savev2_quant_conv2d_kernel_min_read_readvariableop2savev2_quant_conv2d_kernel_max_read_readvariableop;savev2_quant_conv2d_post_activation_min_read_readvariableop;savev2_quant_conv2d_post_activation_max_read_readvariableop8savev2_quant_conv2d_1_optimizer_step_read_readvariableop4savev2_quant_conv2d_1_kernel_min_read_readvariableop4savev2_quant_conv2d_1_kernel_max_read_readvariableop=savev2_quant_conv2d_1_post_activation_min_read_readvariableop=savev2_quant_conv2d_1_post_activation_max_read_readvariableop8savev2_quant_conv2d_2_optimizer_step_read_readvariableop4savev2_quant_conv2d_2_kernel_min_read_readvariableop4savev2_quant_conv2d_2_kernel_max_read_readvariableop=savev2_quant_conv2d_2_post_activation_min_read_readvariableop=savev2_quant_conv2d_2_post_activation_max_read_readvariableop8savev2_quant_conv2d_3_optimizer_step_read_readvariableop4savev2_quant_conv2d_3_kernel_min_read_readvariableop4savev2_quant_conv2d_3_kernel_max_read_readvariableop=savev2_quant_conv2d_3_post_activation_min_read_readvariableop=savev2_quant_conv2d_3_post_activation_max_read_readvariableop8savev2_quant_conv2d_4_optimizer_step_read_readvariableop4savev2_quant_conv2d_4_kernel_min_read_readvariableop4savev2_quant_conv2d_4_kernel_max_read_readvariableop=savev2_quant_conv2d_4_post_activation_min_read_readvariableop=savev2_quant_conv2d_4_post_activation_max_read_readvariableop8savev2_quant_conv2d_5_optimizer_step_read_readvariableop4savev2_quant_conv2d_5_kernel_min_read_readvariableop4savev2_quant_conv2d_5_kernel_max_read_readvariableop=savev2_quant_conv2d_5_post_activation_min_read_readvariableop=savev2_quant_conv2d_5_post_activation_max_read_readvariableop8savev2_quant_conv2d_6_optimizer_step_read_readvariableop4savev2_quant_conv2d_6_kernel_min_read_readvariableop4savev2_quant_conv2d_6_kernel_max_read_readvariableop=savev2_quant_conv2d_6_post_activation_min_read_readvariableop=savev2_quant_conv2d_6_post_activation_max_read_readvariableop3savev2_quant_add_optimizer_step_read_readvariableop/savev2_quant_add_output_min_read_readvariableop/savev2_quant_add_output_max_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ë
_input_shapes¹
¶: : : : : ::: : : ::: : : ::: : : ::: : : ::: : : ::: : : ::: : : : : ::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: : 


_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: : #

_output_shapes
:: $

_output_shapes
::%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
:: 7

_output_shapes
::8

_output_shapes
: 
Ñ$
¥
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_6978

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó


#__inference_abpn_layer_call_fn_4207
input_1
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: #
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: $

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: $

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: $

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: $

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35: 

unknown_36: $

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41: 

unknown_42: 

unknown_43: 

unknown_44: 
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_abpn_layer_call_and_return_conditional_losses_4112
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
´
^
B__inference_lambda_2_layer_call_and_return_conditional_losses_4223

inputs
identity\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
clip_by_value/MinimumMinimuminputs clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿs
IdentityIdentityclip_by_value:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï	
þ
-__inference_quant_conv2d_2_layer_call_fn_6645

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_4720
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï$
£
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_3843

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0¡
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²	

@__inference_lambda_layer_call_and_return_conditional_losses_4396

inputs
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
concatConcatV2inputsinputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*É
_input_shapes·
´:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
^
B__inference_lambda_2_layer_call_and_return_conditional_losses_7259

inputs
identity\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
clip_by_value/MinimumMinimuminputs clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿs
IdentityIdentityclip_by_value:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã#
¥
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_7115

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity¢BiasAdd/ReadVariableOp¢?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1¢ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2¢8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp¢:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ð
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype0È
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype0ð
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(È
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0¶
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"âL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ß
serving_defaultË
U
input_1J
serving_default_input_1:0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿV
lambda_2J
StatefulPartitionedCall:0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¡
ê
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer-11
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
ý
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
quantize_layer_min
quantize_layer_max
quantizer_vars
optimizer_step"
_tf_keras_layer
Û
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
	&layer
'optimizer_step
(_weight_vars
)
kernel_min
*
kernel_max
+_quantize_activations
,post_activation_min
-post_activation_max
._output_quantizers"
_tf_keras_layer
Û
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
	5layer
6optimizer_step
7_weight_vars
8
kernel_min
9
kernel_max
:_quantize_activations
;post_activation_min
<post_activation_max
=_output_quantizers"
_tf_keras_layer
Û
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
	Dlayer
Eoptimizer_step
F_weight_vars
G
kernel_min
H
kernel_max
I_quantize_activations
Jpost_activation_min
Kpost_activation_max
L_output_quantizers"
_tf_keras_layer
Û
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
	Slayer
Toptimizer_step
U_weight_vars
V
kernel_min
W
kernel_max
X_quantize_activations
Ypost_activation_min
Zpost_activation_max
[_output_quantizers"
_tf_keras_layer
Û
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
	blayer
coptimizer_step
d_weight_vars
e
kernel_min
f
kernel_max
g_quantize_activations
hpost_activation_min
ipost_activation_max
j_output_quantizers"
_tf_keras_layer
Û
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
	qlayer
roptimizer_step
s_weight_vars
t
kernel_min
u
kernel_max
v_quantize_activations
wpost_activation_min
xpost_activation_max
y_output_quantizers"
_tf_keras_layer
¥
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

layer
optimizer_step
_weight_vars

kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers"
_tf_keras_layer
Ó
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers

output_min

output_max
_output_quantizer_vars"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
 	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses"
_tf_keras_layer
«
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
0
1
2
©3
ª4
'5
)6
*7
,8
-9
«10
¬11
612
813
914
;15
<16
­17
®18
E19
G20
H21
J22
K23
¯24
°25
T26
V27
W28
Y29
Z30
±31
²32
c33
e34
f35
h36
i37
³38
´39
r40
t41
u42
w43
x44
µ45
¶46
47
48
49
50
51
52
53
54"
trackable_list_wrapper

©0
ª1
«2
¬3
­4
®5
¯6
°7
±8
²9
³10
´11
µ12
¶13"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ê
¼trace_0
½trace_1
¾trace_2
¿trace_32×
#__inference_abpn_layer_call_fn_4207
#__inference_abpn_layer_call_fn_5754
#__inference_abpn_layer_call_fn_5851
#__inference_abpn_layer_call_fn_5342À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¼trace_0z½trace_1z¾trace_2z¿trace_3
¶
Àtrace_0
Átrace_1
Âtrace_2
Ãtrace_32Ã
>__inference_abpn_layer_call_and_return_conditional_losses_5991
>__inference_abpn_layer_call_and_return_conditional_losses_6355
>__inference_abpn_layer_call_and_return_conditional_losses_5450
>__inference_abpn_layer_call_and_return_conditional_losses_5558À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÀtrace_0zÁtrace_1zÂtrace_2zÃtrace_3
ÊBÇ
__inference__wrapped_model_3800input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
Äserving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ð
Êtrace_0
Ëtrace_12
-__inference_quantize_layer_layer_call_fn_6364
-__inference_quantize_layer_layer_call_fn_6373´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÊtrace_0zËtrace_1

Ìtrace_0
Ítrace_12Ë
H__inference_quantize_layer_layer_call_and_return_conditional_losses_6382
H__inference_quantize_layer_layer_call_and_return_conditional_losses_6403´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÌtrace_0zÍtrace_1
):' 2!quantize_layer/quantize_layer_min
):' 2!quantize_layer/quantize_layer_max
:
min_var
max_var"
trackable_dict_wrapper
%:# 2quantize_layer/optimizer_step
S
©0
ª1
'2
)3
*4
,5
-6"
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
Ò
Ótrace_0
Ôtrace_12
+__inference_quant_conv2d_layer_call_fn_6420
+__inference_quant_conv2d_layer_call_fn_6437º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÓtrace_0zÔtrace_1

Õtrace_0
Ötrace_12Í
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_6458
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_6507º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÕtrace_0zÖtrace_1
æ
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses
©kernel
	ªbias
!Ý_jit_compiled_convolution_op"
_tf_keras_layer
#:! 2quant_conv2d/optimizer_step
(
Þ0"
trackable_list_wrapper
#:!2quant_conv2d/kernel_min
#:!2quant_conv2d/kernel_max
 "
trackable_list_wrapper
(:& 2 quant_conv2d/post_activation_min
(:& 2 quant_conv2d/post_activation_max
 "
trackable_list_wrapper
S
«0
¬1
62
83
94
;5
<6"
trackable_list_wrapper
0
«0
¬1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
Ö
ätrace_0
åtrace_12
-__inference_quant_conv2d_1_layer_call_fn_6524
-__inference_quant_conv2d_1_layer_call_fn_6541º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zätrace_0zåtrace_1

ætrace_0
çtrace_12Ñ
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_6562
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_6611º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zætrace_0zçtrace_1
æ
è	variables
étrainable_variables
êregularization_losses
ë	keras_api
ì__call__
+í&call_and_return_all_conditional_losses
«kernel
	¬bias
!î_jit_compiled_convolution_op"
_tf_keras_layer
%:# 2quant_conv2d_1/optimizer_step
(
ï0"
trackable_list_wrapper
%:#2quant_conv2d_1/kernel_min
%:#2quant_conv2d_1/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_1/post_activation_min
*:( 2"quant_conv2d_1/post_activation_max
 "
trackable_list_wrapper
S
­0
®1
E2
G3
H4
J5
K6"
trackable_list_wrapper
0
­0
®1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
Ö
õtrace_0
ötrace_12
-__inference_quant_conv2d_2_layer_call_fn_6628
-__inference_quant_conv2d_2_layer_call_fn_6645º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zõtrace_0zötrace_1

÷trace_0
øtrace_12Ñ
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_6666
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_6715º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z÷trace_0zøtrace_1
æ
ù	variables
útrainable_variables
ûregularization_losses
ü	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses
­kernel
	®bias
!ÿ_jit_compiled_convolution_op"
_tf_keras_layer
%:# 2quant_conv2d_2/optimizer_step
(
0"
trackable_list_wrapper
%:#2quant_conv2d_2/kernel_min
%:#2quant_conv2d_2/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_2/post_activation_min
*:( 2"quant_conv2d_2/post_activation_max
 "
trackable_list_wrapper
S
¯0
°1
T2
V3
W4
Y5
Z6"
trackable_list_wrapper
0
¯0
°1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
Ö
trace_0
trace_12
-__inference_quant_conv2d_3_layer_call_fn_6732
-__inference_quant_conv2d_3_layer_call_fn_6749º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ñ
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_6770
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_6819º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
æ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
¯kernel
	°bias
!_jit_compiled_convolution_op"
_tf_keras_layer
%:# 2quant_conv2d_3/optimizer_step
(
0"
trackable_list_wrapper
%:#2quant_conv2d_3/kernel_min
%:#2quant_conv2d_3/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_3/post_activation_min
*:( 2"quant_conv2d_3/post_activation_max
 "
trackable_list_wrapper
S
±0
²1
c2
e3
f4
h5
i6"
trackable_list_wrapper
0
±0
²1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
Ö
trace_0
trace_12
-__inference_quant_conv2d_4_layer_call_fn_6836
-__inference_quant_conv2d_4_layer_call_fn_6853º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ñ
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_6874
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_6923º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
æ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
±kernel
	²bias
!¡_jit_compiled_convolution_op"
_tf_keras_layer
%:# 2quant_conv2d_4/optimizer_step
(
¢0"
trackable_list_wrapper
%:#2quant_conv2d_4/kernel_min
%:#2quant_conv2d_4/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_4/post_activation_min
*:( 2"quant_conv2d_4/post_activation_max
 "
trackable_list_wrapper
S
³0
´1
r2
t3
u4
w5
x6"
trackable_list_wrapper
0
³0
´1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
Ö
¨trace_0
©trace_12
-__inference_quant_conv2d_5_layer_call_fn_6940
-__inference_quant_conv2d_5_layer_call_fn_6957º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¨trace_0z©trace_1

ªtrace_0
«trace_12Ñ
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_6978
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_7027º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zªtrace_0z«trace_1
æ
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses
³kernel
	´bias
!²_jit_compiled_convolution_op"
_tf_keras_layer
%:# 2quant_conv2d_5/optimizer_step
(
³0"
trackable_list_wrapper
%:#2quant_conv2d_5/kernel_min
%:#2quant_conv2d_5/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_5/post_activation_min
*:( 2"quant_conv2d_5/post_activation_max
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ì
¹trace_0
ºtrace_12
%__inference_lambda_layer_call_fn_7035
%__inference_lambda_layer_call_fn_7043À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¹trace_0zºtrace_1

»trace_0
¼trace_12Ç
@__inference_lambda_layer_call_and_return_conditional_losses_7052
@__inference_lambda_layer_call_and_return_conditional_losses_7061À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z»trace_0z¼trace_1
X
µ0
¶1
2
3
4
5
6"
trackable_list_wrapper
0
µ0
¶1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö
Âtrace_0
Ãtrace_12
-__inference_quant_conv2d_6_layer_call_fn_7078
-__inference_quant_conv2d_6_layer_call_fn_7095º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÂtrace_0zÃtrace_1

Ätrace_0
Åtrace_12Ñ
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_7115
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_7163º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÄtrace_0zÅtrace_1
æ
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses
µkernel
	¶bias
!Ì_jit_compiled_convolution_op"
_tf_keras_layer
%:# 2quant_conv2d_6/optimizer_step
(
Í0"
trackable_list_wrapper
%:#2quant_conv2d_6/kernel_min
%:#2quant_conv2d_6/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_6/post_activation_min
*:( 2"quant_conv2d_6/post_activation_max
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ì
Ótrace_0
Ôtrace_12
(__inference_quant_add_layer_call_fn_7173
(__inference_quant_add_layer_call_fn_7183º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÓtrace_0zÔtrace_1

Õtrace_0
Ötrace_12Ç
C__inference_quant_add_layer_call_and_return_conditional_losses_7194
C__inference_quant_add_layer_call_and_return_conditional_losses_7221º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÕtrace_0zÖtrace_1
«
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"
_tf_keras_layer
 : 2quant_add/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
: 2quant_add/output_min
: 2quant_add/output_max
<
min_var
max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
	variables
trainable_variables
regularization_losses
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
Ð
âtrace_0
ãtrace_12
'__inference_lambda_1_layer_call_fn_7226
'__inference_lambda_1_layer_call_fn_7231À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zâtrace_0zãtrace_1

ätrace_0
åtrace_12Ë
B__inference_lambda_1_layer_call_and_return_conditional_losses_7236
B__inference_lambda_1_layer_call_and_return_conditional_losses_7241À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zätrace_0zåtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
£	variables
¤trainable_variables
¥regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
Ð
ëtrace_0
ìtrace_12
'__inference_lambda_2_layer_call_fn_7246
'__inference_lambda_2_layer_call_fn_7251À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zëtrace_0zìtrace_1

ítrace_0
îtrace_12Ë
B__inference_lambda_2_layer_call_and_return_conditional_losses_7259
B__inference_lambda_2_layer_call_and_return_conditional_losses_7267À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zítrace_0zîtrace_1
':%2conv2d/kernel
:2conv2d/bias
):'2conv2d_1/kernel
:2conv2d_1/bias
):'2conv2d_2/kernel
:2conv2d_2/bias
):'2conv2d_3/kernel
:2conv2d_3/bias
):'2conv2d_4/kernel
:2conv2d_4/bias
):'2conv2d_5/kernel
:2conv2d_5/bias
):'2conv2d_6/kernel
:2conv2d_6/bias
æ
0
1
2
'3
)4
*5
,6
-7
68
89
910
;11
<12
E13
G14
H15
J16
K17
T18
V19
W20
Y21
Z22
c23
e24
f25
h26
i27
r28
t29
u30
w31
x32
33
34
35
36
37
38
39
40"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
öBó
#__inference_abpn_layer_call_fn_4207input_1"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
õBò
#__inference_abpn_layer_call_fn_5754inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
õBò
#__inference_abpn_layer_call_fn_5851inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
öBó
#__inference_abpn_layer_call_fn_5342input_1"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
>__inference_abpn_layer_call_and_return_conditional_losses_5991inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
>__inference_abpn_layer_call_and_return_conditional_losses_6355inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
>__inference_abpn_layer_call_and_return_conditional_losses_5450input_1"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
>__inference_abpn_layer_call_and_return_conditional_losses_5558input_1"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÉBÆ
"__inference_signature_wrapper_5657input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
óBð
-__inference_quantize_layer_layer_call_fn_6364inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
óBð
-__inference_quantize_layer_layer_call_fn_6373inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quantize_layer_layer_call_and_return_conditional_losses_6382inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quantize_layer_layer_call_and_return_conditional_losses_6403inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
C
'0
)1
*2
,3
-4"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
÷Bô
+__inference_quant_conv2d_layer_call_fn_6420inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
÷Bô
+__inference_quant_conv2d_layer_call_fn_6437inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_6458inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_6507inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
(
ª0"
trackable_list_wrapper
(
ª0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
©0
ô2"
trackable_tuple_wrapper
C
60
81
92
;3
<4"
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
-__inference_quant_conv2d_1_layer_call_fn_6524inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ùBö
-__inference_quant_conv2d_1_layer_call_fn_6541inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_6562inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_6611inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
(
¬0"
trackable_list_wrapper
(
¬0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
è	variables
étrainable_variables
êregularization_losses
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
«0
ú2"
trackable_tuple_wrapper
C
E0
G1
H2
J3
K4"
trackable_list_wrapper
'
D0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
-__inference_quant_conv2d_2_layer_call_fn_6628inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ùBö
-__inference_quant_conv2d_2_layer_call_fn_6645inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_6666inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_6715inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
(
®0"
trackable_list_wrapper
(
®0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
ù	variables
útrainable_variables
ûregularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
­0
2"
trackable_tuple_wrapper
C
T0
V1
W2
Y3
Z4"
trackable_list_wrapper
'
S0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
-__inference_quant_conv2d_3_layer_call_fn_6732inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ùBö
-__inference_quant_conv2d_3_layer_call_fn_6749inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_6770inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_6819inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
(
°0"
trackable_list_wrapper
(
°0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
¯0
2"
trackable_tuple_wrapper
C
c0
e1
f2
h3
i4"
trackable_list_wrapper
'
b0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
-__inference_quant_conv2d_4_layer_call_fn_6836inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ùBö
-__inference_quant_conv2d_4_layer_call_fn_6853inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_6874inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_6923inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
(
²0"
trackable_list_wrapper
(
²0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
±0
2"
trackable_tuple_wrapper
C
r0
t1
u2
w3
x4"
trackable_list_wrapper
'
q0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
-__inference_quant_conv2d_5_layer_call_fn_6940inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ùBö
-__inference_quant_conv2d_5_layer_call_fn_6957inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_6978inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_7027inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
(
´0"
trackable_list_wrapper
(
´0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
³0
2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
%__inference_lambda_layer_call_fn_7035inputs/0inputs/1inputs/2inputs/3"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
%__inference_lambda_layer_call_fn_7043inputs/0inputs/1inputs/2inputs/3"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
²B¯
@__inference_lambda_layer_call_and_return_conditional_losses_7052inputs/0inputs/1inputs/2inputs/3"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
²B¯
@__inference_lambda_layer_call_and_return_conditional_losses_7061inputs/0inputs/1inputs/2inputs/3"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
H
0
1
2
3
4"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
-__inference_quant_conv2d_6_layer_call_fn_7078inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ùBö
-__inference_quant_conv2d_6_layer_call_fn_7095inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_7115inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_7163inputs"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
(
¶0"
trackable_list_wrapper
(
¶0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
1
µ0
2"
trackable_tuple_wrapper
8
0
1
2"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bý
(__inference_quant_add_layer_call_fn_7173inputs/0inputs/1"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bý
(__inference_quant_add_layer_call_fn_7183inputs/0inputs/1"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_quant_add_layer_call_and_return_conditional_losses_7194inputs/0inputs/1"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_quant_add_layer_call_and_return_conditional_losses_7221inputs/0inputs/1"º
±²­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
'__inference_lambda_1_layer_call_fn_7226inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ùBö
'__inference_lambda_1_layer_call_fn_7231inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
B__inference_lambda_1_layer_call_and_return_conditional_losses_7236inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
B__inference_lambda_1_layer_call_and_return_conditional_losses_7241inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
'__inference_lambda_2_layer_call_fn_7246inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ùBö
'__inference_lambda_2_layer_call_fn_7251inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
B__inference_lambda_2_layer_call_and_return_conditional_losses_7259inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
B__inference_lambda_2_layer_call_and_return_conditional_losses_7267inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
)min_var
*max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
8min_var
9max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
Gmin_var
Hmax_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
Vmin_var
Wmax_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
emin_var
fmax_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
tmin_var
umax_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
min_var
max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
__inference__wrapped_model_3800ßB©)*ª,-«89¬;<­GH®JK¯VW°YZ±ef²hi³tu´wxµ¶J¢G
@¢=
;8
input_1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "MªJ
H
lambda_2<9
lambda_2+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
>__inference_abpn_layer_call_and_return_conditional_losses_5450ÙB©)*ª,-«89¬;<­GH®JK¯VW°YZ±ef²hi³tu´wxµ¶R¢O
H¢E
;8
input_1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
>__inference_abpn_layer_call_and_return_conditional_losses_5558ÙB©)*ª,-«89¬;<­GH®JK¯VW°YZ±ef²hi³tu´wxµ¶R¢O
H¢E
;8
input_1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
>__inference_abpn_layer_call_and_return_conditional_losses_5991ØB©)*ª,-«89¬;<­GH®JK¯VW°YZ±ef²hi³tu´wxµ¶Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
>__inference_abpn_layer_call_and_return_conditional_losses_6355ØB©)*ª,-«89¬;<­GH®JK¯VW°YZ±ef²hi³tu´wxµ¶Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ô
#__inference_abpn_layer_call_fn_4207ÌB©)*ª,-«89¬;<­GH®JK¯VW°YZ±ef²hi³tu´wxµ¶R¢O
H¢E
;8
input_1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
#__inference_abpn_layer_call_fn_5342ÌB©)*ª,-«89¬;<­GH®JK¯VW°YZ±ef²hi³tu´wxµ¶R¢O
H¢E
;8
input_1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
#__inference_abpn_layer_call_fn_5754ËB©)*ª,-«89¬;<­GH®JK¯VW°YZ±ef²hi³tu´wxµ¶Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
#__inference_abpn_layer_call_fn_5851ËB©)*ª,-«89¬;<­GH®JK¯VW°YZ±ef²hi³tu´wxµ¶Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÛ
B__inference_lambda_1_layer_call_and_return_conditional_losses_7236Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Û
B__inference_lambda_1_layer_call_and_return_conditional_losses_7241Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
'__inference_lambda_1_layer_call_fn_7226Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
'__inference_lambda_1_layer_call_fn_7231Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÛ
B__inference_lambda_2_layer_call_and_return_conditional_losses_7259Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Û
B__inference_lambda_2_layer_call_and_return_conditional_losses_7267Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
'__inference_lambda_2_layer_call_fn_7246Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
'__inference_lambda_2_layer_call_fn_7251Q¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
@__inference_lambda_layer_call_and_return_conditional_losses_7052Û¢
¢
üø
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/2+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/3+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
  
@__inference_lambda_layer_call_and_return_conditional_losses_7061Û¢
¢
üø
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/2+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/3+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ø
%__inference_lambda_layer_call_fn_7035Î¢
¢
üø
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/2+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/3+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿø
%__inference_lambda_layer_call_fn_7043Î¢
¢
üø
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/2+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/3+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
C__inference_quant_add_layer_call_and_return_conditional_losses_7194ß¢
¢
|
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 §
C__inference_quant_add_layer_call_and_return_conditional_losses_7221ß¢
¢
|
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ÿ
(__inference_quant_add_layer_call_fn_7173Ò¢
¢
|
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
(__inference_quant_add_layer_call_fn_7183Ò¢
¢
|
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<9
inputs/1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_6562«89¬;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ç
H__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_6611«89¬;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
-__inference_quant_conv2d_1_layer_call_fn_6524«89¬;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
-__inference_quant_conv2d_1_layer_call_fn_6541«89¬;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_6666­GH®JKM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ç
H__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_6715­GH®JKM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
-__inference_quant_conv2d_2_layer_call_fn_6628­GH®JKM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
-__inference_quant_conv2d_2_layer_call_fn_6645­GH®JKM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_6770¯VW°YZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ç
H__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_6819¯VW°YZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
-__inference_quant_conv2d_3_layer_call_fn_6732¯VW°YZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
-__inference_quant_conv2d_3_layer_call_fn_6749¯VW°YZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_6874±ef²hiM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ç
H__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_6923±ef²hiM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
-__inference_quant_conv2d_4_layer_call_fn_6836±ef²hiM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
-__inference_quant_conv2d_4_layer_call_fn_6853±ef²hiM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_6978³tu´wxM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ç
H__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_7027³tu´wxM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
-__inference_quant_conv2d_5_layer_call_fn_6940³tu´wxM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
-__inference_quant_conv2d_5_layer_call_fn_6957³tu´wxM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_7115µ¶M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ë
H__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_7163µ¶M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_quant_conv2d_6_layer_call_fn_7078µ¶M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
-__inference_quant_conv2d_6_layer_call_fn_7095µ¶M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿå
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_6458©)*ª,-M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 å
F__inference_quant_conv2d_layer_call_and_return_conditional_losses_6507©)*ª,-M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ½
+__inference_quant_conv2d_layer_call_fn_6420©)*ª,-M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
+__inference_quant_conv2d_layer_call_fn_6437©)*ª,-M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
H__inference_quantize_layer_layer_call_and_return_conditional_losses_6382M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 á
H__inference_quantize_layer_layer_call_and_return_conditional_losses_6403M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¹
-__inference_quantize_layer_layer_call_fn_6364M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
-__inference_quantize_layer_layer_call_fn_6373M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"__inference_signature_wrapper_5657êB©)*ª,-«89¬;<­GH®JK¯VW°YZ±ef²hi³tu´wxµ¶U¢R
¢ 
KªH
F
input_1;8
input_1+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"MªJ
H
lambda_2<9
lambda_2+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ