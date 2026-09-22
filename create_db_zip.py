import os
import zipfile
import time

def make_db_zip(source_dir="./pdf_db", output_zip="PDF_db.zip"):
    if not os.path.exists(source_dir):
        print(f"[ERROR] Source directory '{source_dir}' not found!")
        return

    print("="*60)
    print("  OmniDoc AI -- Compressing Vector Database")
    print("="*60)
    print(f"Source folder: {source_dir}")
    print(f"Output file:   {output_zip}\n")

    start_time=time.time()
    file_list=[]
    for root,dirs,files in os.walk(source_dir):
        for f in files:
            file_list.append(os.path.join(root, f))

    total_files=len(file_list)
    print(f"Found {total_files} files to compress...")

    with zipfile.ZipFile(output_zip,"w",zipfile.ZIP_DEFLATED) as zf:
        for idx, file_path in enumerate(file_list,1):
            arcname=os.path.relpath(file_path,".")
            zf.write(file_path,arcname)
            print(f"  [{idx}/{total_files}] Added {os.path.basename(file_path)}", end="\r")

    elapsed=time.time()-start_time
    zip_size_mb=os.path.getsize(output_zip)/(1024*1024)

    print(f"\n\n[SUCCESS] '{output_zip}' created successfully in {elapsed:.1f}s!")
    print(f"Compressed size: {zip_size_mb:.2f} MB")
    print("\nNext Steps:")
    print("1. Upload 'PDF_db.zip' to Google Drive (set access to 'Anyone with link')")
    print("   OR upload as a Release Asset in your GitHub Repository.")
    print("2. Copy the share link.")
    print("3. Add to Streamlit Cloud Secrets:")
    print("   VECTOR_DB_URL = \"https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing\"")
    print("="*60)

if __name__=="__main__":
    make_db_zip()