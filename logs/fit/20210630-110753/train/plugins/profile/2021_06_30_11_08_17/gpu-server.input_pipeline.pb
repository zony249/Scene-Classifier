	??Mo@??Mo@!??Mo@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??Mo@??????@1-=??Ikn@AW??????IN?E????rEagerKernelExecute 0*	?C?l???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorA???F:A@!O?&???X@)A???F:A@1O?&???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?IӠh??!jZ6?y]??)?IӠh??1jZ6?y]??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism㪲?????!???ט??)H¾?D???1?lqܷצ?:Preprocessing2F
Iterator::Model?Χ?U??!7?q????)?ҥI*s?1,t???ǋ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?<֌:A@!??#[?X@)G???R{a?1E?N@?Wy?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI Ia?b=??Q?zBt
?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????@??????@!??????@      ??!       "	-=??Ikn@-=??Ikn@!-=??Ikn@*      ??!       2	W??????W??????!W??????:	N?E????N?E????!N?E????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q Ia?b=??y?zBt
?X@