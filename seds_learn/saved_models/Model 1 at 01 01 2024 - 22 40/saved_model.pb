��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
�
VariableVarHandleOp*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:
*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:
*
dtype0
�

Variable_1VarHandleOp*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape
:@
*
shared_name
Variable_1
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:@
*
dtype0
�

Variable_2VarHandleOp*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape:@*
shared_name
Variable_2
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:@*
dtype0
�

Variable_3VarHandleOp*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:	�@*
shared_name
Variable_3
j
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:	�@*
dtype0
�

Variable_4VarHandleOp*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:�*
shared_name
Variable_4
f
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes	
:�*
dtype0
�

Variable_5VarHandleOp*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:
��*
shared_name
Variable_5
k
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5* 
_output_shapes
:
��*
dtype0
�

Variable_6VarHandleOp*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:�*
shared_name
Variable_6
f
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes	
:�*
dtype0
�

Variable_7VarHandleOp*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:
� �*
shared_name
Variable_7
k
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7* 
_output_shapes
:
� �*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
w1
b1
w2
b2
w3
b3
w4
b4
	__call__

accuracy_fn
loss_fn

signatures*
A;
VARIABLE_VALUE
Variable_7w1/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUE
Variable_6b1/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUE
Variable_5w2/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUE
Variable_4b2/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUE
Variable_3w3/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUE
Variable_2b3/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUE
Variable_1w4/.ATTRIBUTES/VARIABLE_VALUE*
?9
VARIABLE_VALUEVariableb4/.ATTRIBUTES/VARIABLE_VALUE*

trace_0* 

trace_0* 
)
trace_0
trace_1
trace_2* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filename
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1VariableConst*
Tin
2
*
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
GPU 2J 8� *(
f#R!
__inference__traced_save_180482
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable*
Tin
2	*
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_180515ׯ
�
_
drop_layer_cond_false_180246!
drop_layer_cond_identity_relu
drop_layer_cond_identityv
drop_layer/cond/IdentityIdentitydrop_layer_cond_identity_relu*
T0*(
_output_shapes
:����������"=
drop_layer_cond_identity!drop_layer/cond/Identity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:N J
(
_output_shapes
:����������

_user_specified_nameRelu
�+
B
__inference_loss_fn_180414

logits

labels
identityh
&softmax_cross_entropy_with_logits/RankConst*
_output_shapes
: *
dtype0*
value	B :x
'softmax_cross_entropy_with_logits/ShapeConst*
_output_shapes
:*
dtype0*
valueB"��  
   j
(softmax_cross_entropy_with_logits/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :z
)softmax_cross_entropy_with_logits/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"��  
   i
'softmax_cross_entropy_with_logits/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
%softmax_cross_entropy_with_logits/SubSub1softmax_cross_entropy_with_logits/Rank_1:output:00softmax_cross_entropy_with_logits/Sub/y:output:0*
T0*
_output_shapes
: �
-softmax_cross_entropy_with_logits/Slice/beginPack)softmax_cross_entropy_with_logits/Sub:z:0*
N*
T0*
_output_shapes
:v
,softmax_cross_entropy_with_logits/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:�
'softmax_cross_entropy_with_logits/SliceSlice2softmax_cross_entropy_with_logits/Shape_1:output:06softmax_cross_entropy_with_logits/Slice/begin:output:05softmax_cross_entropy_with_logits/Slice/size:output:0*
Index0*
T0*
_output_shapes
:�
1softmax_cross_entropy_with_logits/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������o
-softmax_cross_entropy_with_logits/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(softmax_cross_entropy_with_logits/concatConcatV2:softmax_cross_entropy_with_logits/concat/values_0:output:00softmax_cross_entropy_with_logits/Slice:output:06softmax_cross_entropy_with_logits/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)softmax_cross_entropy_with_logits/ReshapeReshapelogits1softmax_cross_entropy_with_logits/concat:output:0*
T0* 
_output_shapes
:
��
j
(softmax_cross_entropy_with_logits/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :z
)softmax_cross_entropy_with_logits/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"��  
   k
)softmax_cross_entropy_with_logits/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
'softmax_cross_entropy_with_logits/Sub_1Sub1softmax_cross_entropy_with_logits/Rank_2:output:02softmax_cross_entropy_with_logits/Sub_1/y:output:0*
T0*
_output_shapes
: �
/softmax_cross_entropy_with_logits/Slice_1/beginPack+softmax_cross_entropy_with_logits/Sub_1:z:0*
N*
T0*
_output_shapes
:x
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:�
)softmax_cross_entropy_with_logits/Slice_1Slice2softmax_cross_entropy_with_logits/Shape_2:output:08softmax_cross_entropy_with_logits/Slice_1/begin:output:07softmax_cross_entropy_with_logits/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:�
3softmax_cross_entropy_with_logits/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������q
/softmax_cross_entropy_with_logits/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*softmax_cross_entropy_with_logits/concat_1ConcatV2<softmax_cross_entropy_with_logits/concat_1/values_0:output:02softmax_cross_entropy_with_logits/Slice_1:output:08softmax_cross_entropy_with_logits/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
+softmax_cross_entropy_with_logits/Reshape_1Reshapelabels3softmax_cross_entropy_with_logits/concat_1:output:0*
T0* 
_output_shapes
:
��
�
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits2softmax_cross_entropy_with_logits/Reshape:output:04softmax_cross_entropy_with_logits/Reshape_1:output:0*
T0*(
_output_shapes
:��:
��
k
)softmax_cross_entropy_with_logits/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
'softmax_cross_entropy_with_logits/Sub_2Sub/softmax_cross_entropy_with_logits/Rank:output:02softmax_cross_entropy_with_logits/Sub_2/y:output:0*
T0*
_output_shapes
: y
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: �
.softmax_cross_entropy_with_logits/Slice_2/sizePack+softmax_cross_entropy_with_logits/Sub_2:z:0*
N*
T0*
_output_shapes
:�
)softmax_cross_entropy_with_logits/Slice_2Slice0softmax_cross_entropy_with_logits/Shape:output:08softmax_cross_entropy_with_logits/Slice_2/begin:output:07softmax_cross_entropy_with_logits/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:�
+softmax_cross_entropy_with_logits/Reshape_2Reshape(softmax_cross_entropy_with_logits:loss:02softmax_cross_entropy_with_logits/Slice_2:output:0*
T0*
_output_shapes

:��O
ConstConst*
_output_shapes
:*
dtype0*
valueB: s
MeanMean4softmax_cross_entropy_with_logits/Reshape_2:output:0Const:output:0*
T0*
_output_shapes
: D
IdentityIdentityMean:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:
��
:
��
:HD
 
_output_shapes
:
��

 
_user_specified_namelabels:H D
 
_output_shapes
:
��

 
_user_specified_namelogits
�
j
drop_layer_cond_true_180245*
&drop_layer_cond_dropout_layer_mul_relu
drop_layer_cond_identity�h
#drop_layer/cond/dropout_layer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
!drop_layer/cond/dropout_layer/MulMul&drop_layer_cond_dropout_layer_mul_relu,drop_layer/cond/dropout_layer/Const:output:0*
T0*(
_output_shapes
:�����������
#drop_layer/cond/dropout_layer/ShapeShape&drop_layer_cond_dropout_layer_mul_relu*
T0*
_output_shapes
::���
:drop_layer/cond/dropout_layer/random_uniform/RandomUniformRandomUniform,drop_layer/cond/dropout_layer/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed�q
,drop_layer/cond/dropout_layer/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L?�
*drop_layer/cond/dropout_layer/GreaterEqualGreaterEqualCdrop_layer/cond/dropout_layer/random_uniform/RandomUniform:output:05drop_layer/cond/dropout_layer/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������j
%drop_layer/cond/dropout_layer/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
&drop_layer/cond/dropout_layer/SelectV2SelectV2.drop_layer/cond/dropout_layer/GreaterEqual:z:0%drop_layer/cond/dropout_layer/Mul:z:0.drop_layer/cond/dropout_layer/Const_1:output:0*
T0*(
_output_shapes
:�����������
drop_layer/cond/IdentityIdentity/drop_layer/cond/dropout_layer/SelectV2:output:0*
T0*(
_output_shapes
:����������"=
drop_layer_cond_identity!drop_layer/cond/Identity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:N J
(
_output_shapes
:����������

_user_specified_nameRelu
�G
�
__inference__traced_save_180482
file_prefix5
!read_disablecopyonread_variable_7:
� �2
#read_1_disablecopyonread_variable_6:	�7
#read_2_disablecopyonread_variable_5:
��2
#read_3_disablecopyonread_variable_4:	�6
#read_4_disablecopyonread_variable_3:	�@1
#read_5_disablecopyonread_variable_2:@5
#read_6_disablecopyonread_variable_1:@
/
!read_7_disablecopyonread_variable:

savev2_const
identity_17��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOpw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: s
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_7"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_7^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
� �*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
� �c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
� �w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_variable_6"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_variable_6^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�w
Read_2/DisableCopyOnReadDisableCopyOnRead#read_2_disablecopyonread_variable_5"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp#read_2_disablecopyonread_variable_5^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��w
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_variable_4"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_variable_4^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�w
Read_4/DisableCopyOnReadDisableCopyOnRead#read_4_disablecopyonread_variable_3"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp#read_4_disablecopyonread_variable_3^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@w
Read_5/DisableCopyOnReadDisableCopyOnRead#read_5_disablecopyonread_variable_2"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp#read_5_disablecopyonread_variable_2^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@w
Read_6/DisableCopyOnReadDisableCopyOnRead#read_6_disablecopyonread_variable_1"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp#read_6_disablecopyonread_variable_1^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@
*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@
e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:@
u
Read_7/DisableCopyOnReadDisableCopyOnRead!read_7_disablecopyonread_variable"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp!read_7_disablecopyonread_variable^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	Bw1/.ATTRIBUTES/VARIABLE_VALUEBb1/.ATTRIBUTES/VARIABLE_VALUEBw2/.ATTRIBUTES/VARIABLE_VALUEBb2/.ATTRIBUTES/VARIABLE_VALUEBw3/.ATTRIBUTES/VARIABLE_VALUEBb3/.ATTRIBUTES/VARIABLE_VALUEBw4/.ATTRIBUTES/VARIABLE_VALUEBb4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_16Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_17IdentityIdentity_16:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp*
_output_shapes
 "#
identity_17Identity_17:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
: : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp:=	9

_output_shapes
: 

_user_specified_nameConst:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�'
�
"__inference__traced_restore_180515
file_prefix/
assignvariableop_variable_7:
� �,
assignvariableop_1_variable_6:	�1
assignvariableop_2_variable_5:
��,
assignvariableop_3_variable_4:	�0
assignvariableop_4_variable_3:	�@+
assignvariableop_5_variable_2:@/
assignvariableop_6_variable_1:@
)
assignvariableop_7_variable:


identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	Bw1/.ATTRIBUTES/VARIABLE_VALUEBb1/.ATTRIBUTES/VARIABLE_VALUEBw2/.ATTRIBUTES/VARIABLE_VALUEBb2/.ATTRIBUTES/VARIABLE_VALUEBw3/.ATTRIBUTES/VARIABLE_VALUEBb3/.ATTRIBUTES/VARIABLE_VALUEBw4/.ATTRIBUTES/VARIABLE_VALUEBb4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_7Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_6Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_5Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_4Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_3Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_2Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variableIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
_output_shapes
 "!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�$
�
__inference___call___180284
x
training
2
matmul_readvariableop_resource:
� �*
add_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��,
add_1_readvariableop_resource:	�3
 matmul_2_readvariableop_resource:	�@+
add_2_readvariableop_resource:@2
 matmul_3_readvariableop_resource:@
+
add_3_readvariableop_resource:

identity��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�MatMul_2/ReadVariableOp�MatMul_3/ReadVariableOp�add/ReadVariableOp�add_1/ReadVariableOp�add_2/ReadVariableOp�add_3/ReadVariableOp�drop_layer/condv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
� �*
dtype0e
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������H
ReluReluadd:z:0*
T0*(
_output_shapes
:�����������
drop_layer/condIftrainingRelu:activations:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
else_branch R
drop_layer_cond_false_180246*'
output_shapes
:����������*.
then_branchR
drop_layer_cond_true_180245q
drop_layer/cond/IdentityIdentitydrop_layer/cond:output:0*
T0*(
_output_shapes
:����������z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
MatMul_1MatMul!drop_layer/cond/Identity:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0s
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:����������y
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	�@*
dtype0{
MatMul_2MatMulRelu_1:activations:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@n
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:@*
dtype0r
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@K
Relu_2Relu	add_2:z:0*
T0*'
_output_shapes
:���������@x
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:@
*
dtype0{
MatMul_3MatMulRelu_2:activations:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:
*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^drop_layer/cond*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������� : : : : : : : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2"
drop_layer/conddrop_layer/cond:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:@<

_output_shapes
: 
"
_user_specified_name
training:K G
(
_output_shapes
:���������� 

_user_specified_namex
�+
B
__inference_loss_fn_180375

logits

labels
identityh
&softmax_cross_entropy_with_logits/RankConst*
_output_shapes
: *
dtype0*
value	B :x
'softmax_cross_entropy_with_logits/ShapeConst*
_output_shapes
:*
dtype0*
valueB"�  
   j
(softmax_cross_entropy_with_logits/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :z
)softmax_cross_entropy_with_logits/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"�  
   i
'softmax_cross_entropy_with_logits/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
%softmax_cross_entropy_with_logits/SubSub1softmax_cross_entropy_with_logits/Rank_1:output:00softmax_cross_entropy_with_logits/Sub/y:output:0*
T0*
_output_shapes
: �
-softmax_cross_entropy_with_logits/Slice/beginPack)softmax_cross_entropy_with_logits/Sub:z:0*
N*
T0*
_output_shapes
:v
,softmax_cross_entropy_with_logits/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:�
'softmax_cross_entropy_with_logits/SliceSlice2softmax_cross_entropy_with_logits/Shape_1:output:06softmax_cross_entropy_with_logits/Slice/begin:output:05softmax_cross_entropy_with_logits/Slice/size:output:0*
Index0*
T0*
_output_shapes
:�
1softmax_cross_entropy_with_logits/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������o
-softmax_cross_entropy_with_logits/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(softmax_cross_entropy_with_logits/concatConcatV2:softmax_cross_entropy_with_logits/concat/values_0:output:00softmax_cross_entropy_with_logits/Slice:output:06softmax_cross_entropy_with_logits/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)softmax_cross_entropy_with_logits/ReshapeReshapelogits1softmax_cross_entropy_with_logits/concat:output:0*
T0*
_output_shapes
:	�
j
(softmax_cross_entropy_with_logits/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :z
)softmax_cross_entropy_with_logits/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"�  
   k
)softmax_cross_entropy_with_logits/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
'softmax_cross_entropy_with_logits/Sub_1Sub1softmax_cross_entropy_with_logits/Rank_2:output:02softmax_cross_entropy_with_logits/Sub_1/y:output:0*
T0*
_output_shapes
: �
/softmax_cross_entropy_with_logits/Slice_1/beginPack+softmax_cross_entropy_with_logits/Sub_1:z:0*
N*
T0*
_output_shapes
:x
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:�
)softmax_cross_entropy_with_logits/Slice_1Slice2softmax_cross_entropy_with_logits/Shape_2:output:08softmax_cross_entropy_with_logits/Slice_1/begin:output:07softmax_cross_entropy_with_logits/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:�
3softmax_cross_entropy_with_logits/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������q
/softmax_cross_entropy_with_logits/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*softmax_cross_entropy_with_logits/concat_1ConcatV2<softmax_cross_entropy_with_logits/concat_1/values_0:output:02softmax_cross_entropy_with_logits/Slice_1:output:08softmax_cross_entropy_with_logits/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
+softmax_cross_entropy_with_logits/Reshape_1Reshapelabels3softmax_cross_entropy_with_logits/concat_1:output:0*
T0*
_output_shapes
:	�
�
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits2softmax_cross_entropy_with_logits/Reshape:output:04softmax_cross_entropy_with_logits/Reshape_1:output:0*
T0*&
_output_shapes
:�:	�
k
)softmax_cross_entropy_with_logits/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
'softmax_cross_entropy_with_logits/Sub_2Sub/softmax_cross_entropy_with_logits/Rank:output:02softmax_cross_entropy_with_logits/Sub_2/y:output:0*
T0*
_output_shapes
: y
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: �
.softmax_cross_entropy_with_logits/Slice_2/sizePack+softmax_cross_entropy_with_logits/Sub_2:z:0*
N*
T0*
_output_shapes
:�
)softmax_cross_entropy_with_logits/Slice_2Slice0softmax_cross_entropy_with_logits/Shape:output:08softmax_cross_entropy_with_logits/Slice_2/begin:output:07softmax_cross_entropy_with_logits/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:�
+softmax_cross_entropy_with_logits/Reshape_2Reshape(softmax_cross_entropy_with_logits:loss:02softmax_cross_entropy_with_logits/Slice_2:output:0*
T0*
_output_shapes	
:�O
ConstConst*
_output_shapes
:*
dtype0*
valueB: s
MeanMean4softmax_cross_entropy_with_logits/Reshape_2:output:0Const:output:0*
T0*
_output_shapes
: D
IdentityIdentityMean:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:	�
:	�
:GC

_output_shapes
:	�

 
_user_specified_namelabels:G C

_output_shapes
:	�

 
_user_specified_namelogits
�
F
__inference_accuracy_fn_180297

logits

labels
identityR
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :a
ArgMaxArgMaxlogitsArgMax/dimension:output:0*
T0*#
_output_shapes
:���������T
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
value	B :e
ArgMax_1ArgMaxlabelsArgMax_1/dimension:output:0*
T0*#
_output_shapes
:���������`
EqualEqualArgMax:output:0ArgMax_1:output:0*
T0	*#
_output_shapes
:���������T
CastCast	Equal:z:0*

DstT0*

SrcT0
*#
_output_shapes
:���������O
ConstConst*
_output_shapes
:*
dtype0*
valueB: G
MeanMeanCast:y:0Const:output:0*
T0*
_output_shapes
: D
IdentityIdentityMean:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������
:���������
:OK
'
_output_shapes
:���������

 
_user_specified_namelabels:O K
'
_output_shapes
:���������

 
_user_specified_namelogits
�+
B
__inference_loss_fn_180336

logits

labels
identityh
&softmax_cross_entropy_with_logits/RankConst*
_output_shapes
: *
dtype0*
value	B :x
'softmax_cross_entropy_with_logits/ShapeConst*
_output_shapes
:*
dtype0*
valueB"�  
   j
(softmax_cross_entropy_with_logits/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :z
)softmax_cross_entropy_with_logits/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"�  
   i
'softmax_cross_entropy_with_logits/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
%softmax_cross_entropy_with_logits/SubSub1softmax_cross_entropy_with_logits/Rank_1:output:00softmax_cross_entropy_with_logits/Sub/y:output:0*
T0*
_output_shapes
: �
-softmax_cross_entropy_with_logits/Slice/beginPack)softmax_cross_entropy_with_logits/Sub:z:0*
N*
T0*
_output_shapes
:v
,softmax_cross_entropy_with_logits/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:�
'softmax_cross_entropy_with_logits/SliceSlice2softmax_cross_entropy_with_logits/Shape_1:output:06softmax_cross_entropy_with_logits/Slice/begin:output:05softmax_cross_entropy_with_logits/Slice/size:output:0*
Index0*
T0*
_output_shapes
:�
1softmax_cross_entropy_with_logits/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������o
-softmax_cross_entropy_with_logits/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(softmax_cross_entropy_with_logits/concatConcatV2:softmax_cross_entropy_with_logits/concat/values_0:output:00softmax_cross_entropy_with_logits/Slice:output:06softmax_cross_entropy_with_logits/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)softmax_cross_entropy_with_logits/ReshapeReshapelogits1softmax_cross_entropy_with_logits/concat:output:0*
T0*
_output_shapes
:	�
j
(softmax_cross_entropy_with_logits/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :z
)softmax_cross_entropy_with_logits/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"�  
   k
)softmax_cross_entropy_with_logits/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
'softmax_cross_entropy_with_logits/Sub_1Sub1softmax_cross_entropy_with_logits/Rank_2:output:02softmax_cross_entropy_with_logits/Sub_1/y:output:0*
T0*
_output_shapes
: �
/softmax_cross_entropy_with_logits/Slice_1/beginPack+softmax_cross_entropy_with_logits/Sub_1:z:0*
N*
T0*
_output_shapes
:x
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:�
)softmax_cross_entropy_with_logits/Slice_1Slice2softmax_cross_entropy_with_logits/Shape_2:output:08softmax_cross_entropy_with_logits/Slice_1/begin:output:07softmax_cross_entropy_with_logits/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:�
3softmax_cross_entropy_with_logits/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������q
/softmax_cross_entropy_with_logits/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*softmax_cross_entropy_with_logits/concat_1ConcatV2<softmax_cross_entropy_with_logits/concat_1/values_0:output:02softmax_cross_entropy_with_logits/Slice_1:output:08softmax_cross_entropy_with_logits/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
+softmax_cross_entropy_with_logits/Reshape_1Reshapelabels3softmax_cross_entropy_with_logits/concat_1:output:0*
T0*
_output_shapes
:	�
�
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits2softmax_cross_entropy_with_logits/Reshape:output:04softmax_cross_entropy_with_logits/Reshape_1:output:0*
T0*&
_output_shapes
:�:	�
k
)softmax_cross_entropy_with_logits/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
'softmax_cross_entropy_with_logits/Sub_2Sub/softmax_cross_entropy_with_logits/Rank:output:02softmax_cross_entropy_with_logits/Sub_2/y:output:0*
T0*
_output_shapes
: y
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: �
.softmax_cross_entropy_with_logits/Slice_2/sizePack+softmax_cross_entropy_with_logits/Sub_2:z:0*
N*
T0*
_output_shapes
:�
)softmax_cross_entropy_with_logits/Slice_2Slice0softmax_cross_entropy_with_logits/Shape:output:08softmax_cross_entropy_with_logits/Slice_2/begin:output:07softmax_cross_entropy_with_logits/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:�
+softmax_cross_entropy_with_logits/Reshape_2Reshape(softmax_cross_entropy_with_logits:loss:02softmax_cross_entropy_with_logits/Slice_2:output:0*
T0*
_output_shapes	
:�O
ConstConst*
_output_shapes
:*
dtype0*
valueB: s
MeanMean4softmax_cross_entropy_with_logits/Reshape_2:output:0Const:output:0*
T0*
_output_shapes
: D
IdentityIdentityMean:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:	�
:	�
:GC

_output_shapes
:	�

 
_user_specified_namelabels:G C

_output_shapes
:	�

 
_user_specified_namelogits"�J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:�
�
w1
b1
w2
b2
w3
b3
w4
b4
	__call__

accuracy_fn
loss_fn

signatures"
_generic_user_object
:
� �2Variable
:�2Variable
:
��2Variable
:�2Variable
:	�@2Variable
:@2Variable
:@
2Variable
:
2Variable
�
trace_02�
__inference___call___180284�
���
FullArgSpec
args�
jx

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults
 
annotations� *'�$
����������� 
� 
ztrace_0
�
trace_02�
__inference_accuracy_fn_180297�
���
FullArgSpec
args�
jlogits
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
����������

����������
ztrace_0
�
trace_0
trace_1
trace_22�
__inference_loss_fn_180336
__inference_loss_fn_180375
__inference_loss_fn_180414�
���
FullArgSpec
args�
jlogits
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1ztrace_2
"
signature_map
�B�
__inference___call___180284xtraining"�
���
FullArgSpec
args�
jx

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_accuracy_fn_180297logitslabels"�
���
FullArgSpec
args�
jlogits
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_180336logitslabels"�
���
FullArgSpec
args�
jlogits
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_180375logitslabels"�
���
FullArgSpec
args�
jlogits
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_180414logitslabels"�
���
FullArgSpec
args�
jlogits
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference___call___180284m>�;
4�1
�
x���������� 
�
training 

� "!�
unknown���������
�
__inference_accuracy_fn_180297eQ�N
G�D
 �
logits���������

 �
labels���������

� "�
unknown s
__inference_loss_fn_180336UA�>
7�4
�
logits	�

�
labels	�

� "�
unknown s
__inference_loss_fn_180375UA�>
7�4
�
logits	�

�
labels	�

� "�
unknown u
__inference_loss_fn_180414WC�@
9�6
�
logits
��

�
labels
��

� "�
unknown 