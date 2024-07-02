import streamlit as st
import pandas as pd

# Load CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Save CSV file
def save_data(file_path, df):
    df.to_csv(file_path, index=False)

# Streamlit UI
def main():
    st.title("Feedback Rule")

    file_path = 'user_feedback.csv'
    df = load_data(file_path)

    st.subheader("Current rule")

    # Add a selection column to the dataframe
    df['selected'] = False

    # Editable data table with selection checkboxes
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="data_editor")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save Changes"):
            edited_df = edited_df.drop(columns=['selected'])
            save_data(file_path, edited_df)
            st.success("Changes saved successfully")

    with col2:
        if st.button("Delete Selected Traffic"):
            if not edited_df['selected'].any():
                st.warning("No rules selected for deletion")
            else:
                edited_df = edited_df[edited_df['selected'] == False]
                edited_df = edited_df.drop(columns=['selected'])
                save_data(file_path, edited_df)
                st.success("Selected rows deleted successfully")
                st.experimental_rerun()  # Reload the page to fetch the updated records

if __name__ == "__main__":
    main()
