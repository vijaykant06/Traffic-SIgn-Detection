	?U?&?g@?U?&?g@!?U?&?g@	?pw???pw??!?pw??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?U?&?g@n??E????AgF?Ηg@Y???????rEagerKernelExecute 0*	Vm??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??R{?@!?<?i??X@)??R{?@1?<?i??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch? ?????!K??xE???)? ?????1K??xE???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??b? ̹?!???:????)?????1"?
N??:Preprocessing2F
Iterator::ModelΪ??V???!?`?lώ??)Mjh???1??"sB??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap????@!|fM?ĩX@)???+?s?1?#M9Ú??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?pw??I;?#b|?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	n??E????n??E????!n??E????      ??!       "      ??!       *      ??!       2	gF?Ηg@gF?Ηg@!gF?Ηg@:      ??!       B      ??!       J	??????????????!???????R      ??!       Z	??????????????!???????b      ??!       JCPU_ONLYY?pw??b q;?#b|?X@