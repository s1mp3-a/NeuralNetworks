namespace NeuralNetworks.DataExport;

public static class CsvExporter
{
    public static void ExportCsv(string path, double[] values)
    {
        while (true)
        {
            if (File.Exists(path))
            {
                using (StreamWriter sw = new StreamWriter(path))
                {
                    foreach (var t in values)
                    {
                        sw.WriteLine(t);
                    }
                }
                
            }
            else
            {
                File.Create(path).Close();
                continue;
            }

            break;
        }
    }
}