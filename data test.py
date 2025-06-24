from datetime import date, timedelta, datetime, timezone
from typing import List, Dict, Any
import subprocess
import tarfile
import lz4.frame
import yaml
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

class OoniDataPipeline:
    def __init__(self, bucket: str, subdir: str, aws_cmd: str = r'C:\Program Files\Amazon\AWSCLIV2\aws.exe'):
        """
        1. initializes the AWS S3 bucket
        2. fetches subdirectory info
        3. specify the AWS CLI path
        """
        self.bucket = bucket
        self.subdir = subdir
        self.aws_cmd = aws_cmd

    def list_tar_lz4_files(self, date_subdir: str) -> List[str]:
        """
        1. lists all the .tar.lz4 files in a given S3 path for some specified date
        2. uses path to find all available OONI measurement archives
        """
        s3_path = f's3://{self.bucket}/{self.subdir}/{date_subdir}/'
        cmd = [self.aws_cmd, 's3', 'ls', '--no-sign-request', s3_path]
        try:
            output = subprocess.check_output(cmd, text=True)
            files = []
            for line in output.splitlines():
                if line.strip().endswith('.tar.lz4'):
                    fname = line.split()[-1]
                    files.append(fname)
            return files
        except subprocess.CalledProcessError:
            return []

    def download_file(self, date_subdir: str, filename: str) -> str:
        """
        download the specified file from S3 to the local system
        """
        s3_path = f's3://{self.bucket}/{self.subdir}/{date_subdir}/{filename}'
        cmd = [self.aws_cmd, 's3', 'cp', '--no-sign-request', s3_path, filename]
        subprocess.check_call(cmd)
        return filename

    def extract_yaml_docs_from_lz4tar(self, tar_lz4_path: str) -> List[Dict[str, Any]]:
        """
        1. compresses and extracts applicable YAML documents from the tar-lz4 archive
        2. return yaml documents as a list of python dictionaries
        """
        tar_path = tar_lz4_path[:-4] + '.tar'
        with open(tar_lz4_path, 'rb') as src, open(tar_path, 'wb') as dst:
            dst.write(lz4.frame.decompress(src.read()))
        docs = []
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.yaml'):
                    f = tar.extractfile(member)
                    if f:
                        try:
                            for doc in yaml.safe_load_all(f):
                                docs.append(doc)
                        except Exception:
                            continue
        os.remove(tar_path)
        return docs

    def collect_all_docs(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """
        1. loop through specified date range
        2. gather YAML documents from all available tar-lz4 files
        3. return documents as a list of dictionaries
        """
        docs = []
        current = start_date
        while current <= end_date:
            date_subdir = current.isoformat()
            files = self.list_tar_lz4_files(date_subdir)
            for tf in files:
                print(f"Downloading: {date_subdir}/{tf}")
                self.download_file(date_subdir, tf)
                docs.extend(self.extract_yaml_docs_from_lz4tar(tf))
                os.remove(tf)
            current += timedelta(days=1)
        return docs

    def flatten_ooni_docs(self, docs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        1. convert list of raw YAML documents into dataframe
        2. pull out key fields
        3. handle different data types
        """
        rows = []
        for doc in docs:
            row = {
                'test_name': doc.get('test_name'),
                'probe_cc': doc.get('probe_cc'),
                'probe_asn': doc.get('probe_asn')
            }
            ts = doc.get('test_started')
            if ts is not None:
                try:
                    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                    row['test_started'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                    row['date'] = dt.date()
                except Exception:
                    row['test_started'] = ts
            report = doc.get('report', {})
            if 'transparent_http_proxy' in report:
                row['transparent_http_proxy'] = bool(report['transparent_http_proxy'])
            if isinstance(report.get('requests'), list):
                codes = []
                hosts = []
                for req in report['requests']:
                    resp = req.get('response', {})
                    code = resp.get('code')
                    url = req.get('request', {}).get('url')
                    if url:
                        hosts.append(url)
                    if code:
                        codes.append(code)
                if codes:
                    row['access_success'] = any(c == 200 for c in codes)
                    row['http_response_codes'] = ','.join(str(c) for c in codes)
                if hosts:
                    row['test_url'] = hosts[0]
            if 'queries' in report and isinstance(report['queries'], list):
                resolvers = set()
                for q in report['queries']:
                    r = q.get('resolver')
                    if isinstance(r, list) and len(r) > 0:
                        resolvers.add(str(r[0]))
                row['unique_dns_resolvers'] = len(resolvers) if resolvers else None
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_proxy_detection_by_country(self, df: pd.DataFrame) -> None:
        """
        1. group tests by country
        2. create a bar plot to visualize the fraction of tests that detected a transparent proxy
        """
        if 'probe_cc' in df and 'transparent_http_proxy' in df:
            summary = df.groupby('probe_cc')['transparent_http_proxy'].mean()
            summary = summary[summary > 0].sort_values(ascending=False)
            if len(summary) > 0:
                plt.figure(figsize=(max(7, 0.7*len(summary)), 4))
                plt.bar(summary.index, summary.values, color='purple')
                plt.ylabel('Fraction of Tests With Transparent Proxy')
                plt.xlabel('Country Code')
                plt.title('Transparent Proxy Detected by Country')
                plt.xticks(ticks=range(len(summary)), labels=summary.index, rotation=30, ha='right', fontsize=12)
                plt.tight_layout()
                plt.savefig('ooni_proxy_by_country.png', dpi=140)
                plt.close()

    def plot_website_access_by_day(self, df: pd.DataFrame) -> None:
        """
        1. plot the daily fraction of websites that were accessible
        2. visualize trends over time
        """
        if 'access_success' not in df or 'date' not in df:
            return None
        by_day = df.groupby(['date'])['access_success'].mean()
        if not by_day.empty:
            plt.figure(figsize=(max(8, 1.2*len(by_day)), 4))
            plt.bar(by_day.index.astype(str), by_day.values, color='teal')
            plt.ylabel('Fraction of Sites Accessible')
            plt.xlabel('Date')
            plt.title('Website Access Rate by Date')
            plt.ylim(0,1)
            plt.xticks(rotation=0, fontsize=12)
            plt.tight_layout()
            plt.savefig('ooni_access_by_day.png', dpi=140)
            plt.close()

    def plot_test_type_histogram(self, df: pd.DataFrame) -> None:
        """
        visualize the counts of the most common types of OONI tests performed
        """
        if 'test_name' in df:
            top = df['test_name'].value_counts().head(8)
            plt.figure(figsize=(max(9, 0.85*len(top)), 4))
            plt.bar(top.index, top.values, color='orange')
            plt.ylabel('Count')
            plt.xlabel('Test Type')
            plt.title('Most Common OONI Test Types')
            plt.xticks(ticks=range(len(top)), labels=top.index, rotation=30, ha='right', fontsize=12)
            plt.tight_layout()
            plt.savefig('ooni_test_types.png', dpi=140)
            plt.close()

    def plot_dns_resolver_diversity(self, df: pd.DataFrame) -> None:
        """
        show the distribution of unique DNS resolvers present in test data
        """
        if 'unique_dns_resolvers' in df:
            data = df[df['unique_dns_resolvers'].notnull()]
            if not data.empty:
                max_val = int(data['unique_dns_resolvers'].max())
                plt.figure(figsize=(max(7, 1.2*max_val), 4))
                bins = list(range(1, max_val+2))
                counts, bins, patches = plt.hist(data['unique_dns_resolvers'].astype(int), bins=bins, color='slateblue', rwidth=0.82, align='left')
                plt.xlabel('Unique DNS Resolvers Used')
                plt.ylabel('Number of Tests')
                plt.title('DNS Resolver Diversity Across Tests')
                plt.xticks(ticks=bins[:-1], labels=[str(b) for b in bins[:-1]], rotation=0, fontsize=12)
                plt.tight_layout()
                plt.savefig('ooni_dns_resolvers.png', dpi=140)
                plt.close()

    def to_excel(self, raw_yaml: List[Dict[str, Any]], cleaned_df: pd.DataFrame, chart_paths: List[str], filename: str = 'ooni_results.xlsx') -> None:
        """
        1. export raw data to an excel file
        2. export cleaned data
        3. embed visualizations as images
        """
        images = {}
        for path in chart_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    images[path] = f.read()
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            pd.DataFrame({'raw_yaml': [str(doc) for doc in raw_yaml]}).to_excel(writer, sheet_name='RawYAML', index=False)
            cleaned_df.to_excel(writer, sheet_name='CleanedData', index=False)
            worksheet = writer.sheets['CleanedData']
            for i, col in enumerate(cleaned_df.columns):
                worksheet.set_column(i, i, 18)
            for idx, (path, data) in enumerate(images.items()):
                wsname = f'Chart{idx+1}'
                ws = writer.book.add_worksheet(wsname)
                ws.insert_image('B2', path, {'image_data': BytesIO(data), 'x_scale': 0.92, 'y_scale': 0.92})
        print(f"Excel file saved as {filename}")

if __name__ == "__main__":
    pipeline = OoniDataPipeline(bucket='ooni-data', subdir='canned')
    start, end = date(2012, 12, 5), date(2012, 12, 31)
    docs = pipeline.collect_all_docs(start, end)
    df = pipeline.flatten_ooni_docs(docs)
    pipeline.plot_proxy_detection_by_country(df)
    pipeline.plot_website_access_by_day(df)
    pipeline.plot_test_type_histogram(df)
    pipeline.plot_dns_resolver_diversity(df)
    chart_paths = ['ooni_proxy_by_country.png', 'ooni_access_by_day.png', 'ooni_test_types.png', 'ooni_dns_resolvers.png']
    pipeline.to_excel(raw_yaml=docs, cleaned_df=df, chart_paths=chart_paths, filename='ooni_results.xlsx')
    print("\nAnalysis and export complete. See ooni_results.xlsx.")
