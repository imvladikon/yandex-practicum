<div class="base-markdown base-markdown_with-gallery markdown markdown_size_small markdown_type_theory theory-viewer__markdown task-description__markdown"><div class="paragraph">Perform the exploratory data analysis:</div><ul><li>Look at the sample size.</li><li>Plot the graph of the age distribution in the sample.</li><li>Print 10-15 photos on the screen to check how the dataset works.</li></ul><div class="paragraph">Paths to the files for analysis: '/datasets/faces/labels.csv', '/datasets/faces/final_files/'.</div><div class="paragraph">Provide findings on how the analysis results will affect the model training.</div></div>
<div class="base-markdown base-markdown_with-gallery markdown markdown_size_small markdown_type_theory theory-viewer__markdown notification__content"><div class="paragraph">Here's how you can load the data:</div><pre class="plaintext code-block code-block_theme_light"><div class="code-block__tools"><span class="code-block__clipboard">Copy code</span></div><code class="code-block__code plaintext">labels = pd.read_csv('/datasets/faces/labels.csv')
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        seed=12345) </code></pre></div>