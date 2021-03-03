merged = pd.merge(orig, updated, how='left', left_on='diagnosis', right_on='Field')
merged = merged[['icd9', 'icd10', 'diagnosis', 'stroke_type', 'Updated Stroke type', 'Count']]

def codeStroke(x):
    if type(x['Updated Stroke type']) == float:
        return x['stroke_type']
    if x['Updated Stroke type'] == 'Ischemic Stroke':
        return 'I'
    elif x['Updated Stroke type'] == 'hemorrhagic stroke':
        return 'H'
    elif (x['Updated Stroke type'] == 'Possible Stroke') | (x['Updated Stroke type'] == 'Possible stroke') | (x['Updated Stroke type'] == 'Stroke'):
        return 'P'
    else:
        return ''
merged['stroke_type'] = merged[['stroke_type', 'Updated Stroke type']].apply(lambda x: codeStroke(x), axis=1)
merged.drop(columns=['Updated Stroke type', 'Count'])
merged.to_csv(os.path.join(cfg['WORKING_DATA_DIR'], 'Processed/Manual_Coding/Dx_class_ann_updated.csv'), index=False)