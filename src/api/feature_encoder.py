import pandas as pd


def encode_categorical_values(df, encoder, encoded_cols):
    # Encode using the training columns
    not_keeping_col = list(set(encoded_cols) - set(df.columns))
    # Fill missing columns with 0 to match expected model features.
    # These columns are removed after encoding.
    df[not_keeping_col] = 0

    # Transform via encoder (assumes OneHotEncoder)
    encoded_data = encoder.transform(df[encoded_cols])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(encoded_cols),
        index=df.index,
    )

    # Merge encoded data and drop original columns
    df = df.drop(columns=encoded_cols)
    df = pd.concat([df, encoded_df], axis=1)

    for col in df.columns:
        for not_keep_col in not_keeping_col:
            if col.startswith(not_keep_col):
                df.drop(columns=col, inplace=True)

    return df
