from Bio import SeqIO
import pandas as pd
import os 

# read genome, along with annotated prophage regions (i.e. start, end positions)
# encode
# calc sliding window loss and perplexity
# plot loss/perplexity along genome

from Bio import SeqIO

def extract_phage_genes_by_keywords(gbk_file, keyword_list):
    """
    从GenBank文件中提取注释包含列表中任一关键词的基因或CDS坐标。
    """
    # 将所有关键词转换为小写，方便进行不区分大小写的匹配
    keyword_list_lower = [kw.lower() for kw in keyword_list]

    results = []

    for record in SeqIO.parse(gbk_file, "genbank"):
        # print(f"正在分析基因组: {record.id}")
        found_any = False
        for feature in record.features:
            if feature.type in ["CDS", "gene"]:
                # 合并所有注释文本
                qualifiers = feature.qualifiers
                all_notes = " ".join([
                    qualifiers.get("gene", [""])[0],
                    qualifiers.get("product", [""])[0],
                    qualifiers.get("note", [""])[0],
                    qualifiers.get("function", [""])[0]
                ]).lower()

                # 检查是否包含列表中的任一关键词
                matched_keywords = [kw for kw in keyword_list_lower if kw in all_notes]
                if matched_keywords:
                    start = feature.location.start + 1
                    end = feature.location.end
                    strand = "-" if feature.location.strand == -1 else "+"
                    gene_name = qualifiers.get("gene", ["N/A"])[0]
                    product = qualifiers.get("product", ["N/A"])[0]

                    results.append({
                        "phage name": matched_keywords[0],
                        "gene": gene_name,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "product": product
                    })

                    # print(f"  基因: {gene_name}")
                    # print(f"  坐标: {start} - {end} ({strand}链)")
                    # print(f"  匹配关键词: {', '.join(matched_keywords)}")
                    # print(f"  产物: {product}\n")
                    found_any = True
        
        if not found_any:
            print("  未找到匹配的基因。\n")
    
    return results

def extract_phage_seqs(phage_regions_file, genome_file, output_dir):
    """
    从基因组中提取噬菌体序列，并将其保存为FASTA文件。
    """
    # 读取基因组序列
    genome_record = SeqIO.read(genome_file, "fasta")
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 读取噬菌体区域信息
    phage_regions = pd.read_csv(phage_regions_file)

    for index, row in phage_regions.iterrows():
        phage_name = row['phage name'].replace(" ", "_")  # 替换空格以便于文件命名
        start = row['start'] - 1  # 转换为0-based索引
        end = row['end']
        
        # 提取噬菌体序列
        phage_seq = genome_record.seq[start:end]
        
        # 创建新的SeqRecord对象
        phage_record = SeqIO.SeqRecord(
            phage_seq,
            id=phage_name,
            description=f"Extracted from {genome_record.id}: {start+1}-{end}"
        )
        
        # 保存为FASTA文件
        output_file = os.path.join(output_dir, f"{phage_name}.fasta")
        SeqIO.write(phage_record, output_file, "fasta")
        # print(f"已保存噬菌体序列: {output_file}")

def main():

    my_keywords = [
        "phage SPbeta",      # 特定原噬菌体名称
        "phage PBSX",    # 通用原噬菌体描述
        # "SKIN",       # 噬菌体
        # "SPgama",   # 整合酶，边界标志
        # "transposase", # 转座酶，常与噬菌体相关
        # "extra",       # 像您例子中的 "extrachromosomal"
        # "portal",      # 噬菌体门户蛋白
        # "capsid",      # 噬菌体衣壳蛋白
        # "tail",        # 噬菌体尾部蛋白
    ]

    file_list = [
        "/srv/scratch/z3543429/LLM/TriEvo/dataset/bacteria/NC_000964.3/sequence.gb",
    ]

    for file in file_list:
        print(f"Extracting from: {file}")
        out = os.path.join(os.path.dirname(file), "phage_genes.csv")
        results = pd.DataFrame(extract_phage_genes_by_keywords(file, my_keywords))
        results = results.sort_values(by=['phage name', 'start'])
        results.to_csv(out, index=False)
        phages = pd.DataFrame({
            'phage name': results['phage name'].unique(),
            'start': results.groupby('phage name')['start'].min().values,
            'end': results.groupby('phage name')['end'].max().values
        })
        phages.to_csv(os.path.join(os.path.dirname(file), "phage_regions.csv"), index=False)

        extract_phage_seqs(os.path.join(os.path.dirname(file), "phage_regions.csv"), file, os.path.join(os.path.dirname(file), "phage_seqs"))

if __name__ == "__main__":
    main()