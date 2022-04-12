namespace NeuralNetworks.NeuronWorks;

public class RbNeuron
{
    private readonly double _s;
    private double[] _coefficients;

    internal double Value { get; private set; }

    public RbNeuron(double[] coefficients, double s)
    {
        _coefficients = coefficients;
        _s = s;
    }

    public double Activate(double[] inputs)
    {
        var res = Math.Exp(-(CoeffNorm(inputs) * (0.8236/_s)));
        Value = res;
        return res;
    }

    private double CoeffNorm(double[] inputs)
    {
        var res = 0d;
        
        for (int i = 0; i < _coefficients.Length; i++)
        {
            res += Math.Pow(inputs[i] - _coefficients[i], 2);
        }
        
        return Math.Sqrt(res);
    }
}