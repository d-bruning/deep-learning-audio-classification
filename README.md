# Audio Duration and Genre Classification

## GTZAN Study

Notebook:
gtzan_analysis.ipynb

Source:
src/gtzan/

Results:
results/gtzan/

## FMA Small Study

Notebook:
fma_analysis.ipynb

Source:
src/fma_small/

Results:
results/fma_small/


## GTZAN Results
### Duration Effects

Analysis of the GTZAN dataset demonstrated that audio duration influenced music genre classification performance, although the magnitude of the effect varied substantially across architectures. Several models exhibited large improvements when longer audio excerpts were provided. CapsNet demonstrated the largest increase in performance between one-second and thirty-second excerpts (+23.3%), followed by CNN (+20.9%) and MobileNetV2 (+16.4%). These gains were accompanied by large effect sizes and statistically significant improvements, indicating that additional listening context provided meaningful information for genre recognition.

The duration-performance relationship was non-linear. Performance generally improved as clip duration increased, but the largest gains occurred during the earliest duration transitions. Longer excerpts continued to improve performance for many architectures, although the rate of improvement diminished at larger durations. These findings suggest that additional audio context remains beneficial, but the informational value of each additional second decreases as more context becomes available.

### Architecture Performance

CNN achieved the strongest overall classification performance and produced the highest average accuracy across the evaluated architectures. Performance leadership, however, depended on clip duration. ShallowResNet performed best on one-second excerpts while CNN achieved the strongest performance on thirty-second excerpts. These findings indicate that architecture selection and excerpt duration interact, and that no single architecture uniformly dominates across all experimental conditions.

CRNN delivered the strongest efficiency-to-performance tradeoff, achieving the highest accuracy per million parameters while maintaining competitive classification performance. This result demonstrates that models with relatively modest parameter counts can remain highly effective when their architectural assumptions align with the structure of the problem.

### Genre Effects

Duration sensitivity varied considerably across genres. Classical, Blues, Pop, and Hip-Hop exhibited substantial improvements when longer excerpts were provided, whereas Country, Metal, and Rock showed considerably smaller gains. Rock was particularly notable because performance remained low even for thirty-second excerpts.

These findings suggest that certain genres contain discriminative information that emerges gradually over time, while others can be recognized using relatively short excerpts. The results therefore indicate that duration effects are not uniformly distributed across the genre taxonomy.

### Generalization and Error Analysis

Generalization behavior varied substantially among architectures. Some models achieved strong classification performance at the cost of larger train-validation gaps, whereas others generalized effectively but produced lower overall accuracy. The results demonstrate that predictive performance and generalization quality represent distinct dimensions of model evaluation.

Error analysis revealed recurring confusion patterns across architectures and durations. Certain genre pairs remained difficult to separate even after substantially increasing clip duration, suggesting that some classification errors were driven by overlap in acoustic characteristics rather than insufficient listening context.

## FMA Small Results
### Duration Effects

The FMA Small experiments largely replicated the central findings observed on GTZAN. Increasing audio duration generally improved classification performance, but the magnitude of improvement was substantially smaller than that observed on GTZAN. Duration gains were concentrated primarily in the transition from one-second excerpts to short and moderate-duration excerpts. Beyond approximately five to ten seconds, performance gains became comparatively small.

Unlike GTZAN, where several architectures exhibited dramatic improvements from longer listening windows, FMA Small exhibited more modest gains. MobileNetV2 and CRNN showed the largest duration-related improvements, while CNN demonstrated only limited sensitivity to additional clip length. These findings indicate that duration effects persist on a larger dataset but may be more constrained than initially suggested by GTZAN.

### Architecture Performance

MobileNetV2 achieved the strongest overall performance on FMA Small, producing the highest classification accuracy, strongest F1 score, largest duration gain, and most favorable generalization characteristics. Unlike the GTZAN experiments, where CNN achieved the best overall performance, the FMA results indicate a clear advantage for transfer learning architectures.

CRNN achieved the second strongest overall performance while also exhibiting one of the largest duration-related improvements. Notably, CRNN's learning dynamics differed substantially from MobileNetV2 despite producing similar duration gains, suggesting that multiple architectural approaches can benefit from additional audio context for different reasons.

ViT represented the transformer family within the study. Although ViT did not achieve the highest overall classification performance, it demonstrated relatively strong generalization characteristics and provided an important comparison point against convolutional and recurrent architectures.

### Genre Effects

Genre sensitivity to duration remained highly variable. Folk and Hip-Hop exhibited the largest duration-related recall improvements and accounted for much of the overall benefit derived from longer excerpts. In contrast, Pop and Instrumental demonstrated little or no improvement as clip duration increased.

These findings suggest that duration effects are best understood at the genre level rather than as a universal property of music classification. Certain genres appear to require additional listening context before reliable classification becomes possible, whereas others can be identified using only short excerpts.

### Error Analysis

Error analysis revealed several persistent confusion clusters. The strongest confusions included Hip-Hop ↔ Electronic, Folk ↔ Instrumental, Experimental ↔ Instrumental, and Pop ↔ Rock. These patterns remained present even at longer durations, indicating that additional audio context reduced but did not eliminate several difficult classification boundaries.

Pop proved particularly difficult to classify and exhibited the lowest overall recall among the evaluated genres. The genre was frequently confused with Rock, Folk, Electronic, and International recordings, suggesting that Pop functions as a highly overlapping category within the FMA taxonomy.

### MobileNetV2 Refinement Study

Following completion of the FMA Small experiments, MobileNetV2 was selected for targeted refinement. The architecture was chosen because it achieved the strongest overall performance while simultaneously demonstrating the largest duration effect, strongest generalization behavior, and lowest observed overfitting rate.

The purpose of the refinement phase was not to establish a new benchmark performance level but rather to investigate whether additional optimization could further improve an already successful architecture. This refinement experiment represents a natural continuation of the research process, moving from broad comparison toward focused investigation of the most promising model.

The refinement results should therefore be interpreted separately from the primary comparative study. Whereas the earlier experiments evaluated multiple architectures under standardized conditions, the refinement phase explores how targeted changes affect a single architecture that emerged as particularly effective during the broader evaluation.

### Cross-Dataset Conclusions

Taken together, the GTZAN and FMA Small studies provide strong evidence that audio duration influences music genre classification performance. Importantly, this relationship was observed across multiple datasets and multiple model families, suggesting that duration effects are genuine properties of the classification task rather than artifacts of a particular architecture or dataset.

However, the magnitude of duration effects differed substantially between datasets. Whereas GTZAN exhibited large performance gains from increased listening context, FMA Small demonstrated much smaller improvements and earlier performance plateaus. These findings suggest that dataset characteristics, genre definitions, and training data volume all influence the degree to which additional audio information improves recognition performance.

The analyses further revealed that duration effects were highly genre dependent. In both datasets, a relatively small subset of genres accounted for a disproportionate share of the performance gains associated with longer excerpts. Additional audio context therefore appears to benefit specific classification problems rather than providing uniform improvements across all musical styles.

Finally, architecture selection influenced both overall performance and duration sensitivity. MobileNetV2 emerged as the strongest overall architecture in the FMA experiments, while CNN achieved the strongest performance on GTZAN. At the same time, the existence of duration effects across convolutional, recurrent, transfer-learning, and transformer architectures suggests that clip duration remains an important consideration regardless of the underlying model family.

Collectively, the evidence indicates that audio duration is a meaningful but context-dependent factor in genre classification. Longer excerpts can improve performance, but the magnitude of the benefit depends on the dataset, the genre, and the architecture being used. The results also suggest that substantial gains may be attainable using relatively short clips, with additional listening context producing progressively smaller returns as duration increases.