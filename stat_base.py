"""myStats - python code for the textbook examples"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import norm, t, chi2, mode, binom
import numpy
import scipy.stats as stats
from IPython.display import display, Markdown, Latex
import math
import pandas as pd

class MyStats:
    def __init__(self):
        pass

    def get_pop_sd(arr):

        print("arr : ", arr)
        print("std of arr : ", np.std(arr))

        print("\nMore precision with float32")
        print("std of arr : ", np.std(arr, dtype=np.float32))

        print("\nMore accuracy with float64")
        print("std of arr : ", np.std(arr, dtype=np.float64))

    def get_sample_sd(arr, ddof=1):
        print("arr : ", arr)
        print("std of arr : ", np.std(arr))

        print("\nMore precision with float32")
        print("std of arr : ", np.std(arr, dtype=np.float32, ddof=1))

        print("\nMore accuracy with float64")
        print("std of arr : ", np.std(arr, dtype=np.float64, ddof=1))

    def get_mean(arr):

        mean = scipy.mean(arr)
        print("mean of arr : ",mean)
        return mean

    def get_sum_of_squares(x):
        return sum(map(lambda i: i * i, x))

    def get_sum_of_product(x, y):
        return round(sum([a * b for a, b in zip(x, y)]),2)


    class MeasuresOfCentralTendency:
        def __init__(self):
            pass

        def get_mode(values):
            return mode(values)

        def get_median(values):
            return numpy.median(values)

        def get_mean(values):
            return numpy.mean(values)

        def get_trim_mean(values, proportioncut):
            return scipy.stats.trim_mean(values, proportioncut)

        def get_range(values):
            return max(values) - min(values)

        def get_variance(values,type='sample'):
            if(type == 'sample'):
                return numpy.var(values, ddof=1)
            else:
                return numpy.var(values)

        def get_standard_deviation(values,type='sample'):
            if(type == 'sample'):
                return numpy.std(values, ddof=1)
            else:
                return numpy.std(values)

        def get_coefficient_of_variation(values, type='sample'):
            if (type == 'sample'):
                return round(MyStats.MeasuresOfCentralTendency.get_standard_deviation(values, 'sample'), 2) / round(MyStats.MeasuresOfCentralTendency.get_mean(values),2)
            else:
                return round(MyStats.MeasuresOfCentralTendency.get_standard_deviation(values, 'pop'), 2) / round(MyStats.MeasuresOfCentralTendency.get_mean(values),2)

        def description(array_values):
            array_values.sort()
            d = {'one': array_values}
            df = pd.DataFrame(d)

            if (len(array_values) % 2) == 0:
                qrange = int(len(array_values) / 2)
            else:
                qrange = math.floor(len(array_values) / 2)

            dq1 = {'one': array_values[:qrange]}
            dfq1 = pd.DataFrame(dq1)
            dq2 = {'one': array_values[qrange:]}
            dfq2 = pd.DataFrame(dq2)

            print(df.one.describe())
            print("By the book")
            print('Q1 : %s' % dfq1.one.describe()[5])
            print('Medan : %s' % df.one.describe()[5])
            print('Q3 : %s' % dfq2.one.describe()[5])

        def get_midpoint(value):
            midpoint = value
            if ('-' in str(value)):
                values = value.split('-')
                midpoint = float(values[0]) + (float(values[1]) - float(values[0])) / 2
            return midpoint

        def discrete_population_probability_distribution_describe(values, probs, round_returns=2):

            mean = 0.0
            for i, value in enumerate(values):
                print("%s X %s"%(MyStats.MeasuresOfCentralTendency.get_midpoint(value), probs[i]))
                mean += MyStats.MeasuresOfCentralTendency.get_midpoint(value) * probs[i]

            variance = 0
            for i, value in enumerate(values):
                variance += ((MyStats.MeasuresOfCentralTendency.get_midpoint(value) - mean) ** 2) * probs[i]

            sd = math.sqrt(variance)

            mean = round(mean, 2)
            variance = round(variance, 2)
            sd = round(sd, 2)
            return {'mean': mean, 'variance': variance, 'sd': sd}

    class DiscreteProbabilityDistribution:
        def __init__(self):
            pass

        def get_mu(values, relative_frequencies):
            print("Calculating mean for of a discrete population probability distribution ")
            display(Latex(r'$ \mu = \sum{xP(x)}$'))
            mu = sum([value * relative_frequencies[i] for i, value in enumerate(values)])
            display(Latex(r'$ \mu = %s$'%mu))
            print('****************************')
            return mu

        def get_sd(values, relative_frequencies, round_sd=2):

            print("Calculating standard deviation for of a discrete population probability distribution ")
            mu = MyStats.DiscreteProbabilityDistribution.get_mu(values, relative_frequencies)
            display(Latex(r'$ \sigma = \sqrt{\sum{(x - \mu)^2 P(x)}} $'))
            sd = round(math.sqrt(sum([((value -mu)**2) * relative_frequencies[i] for i, value in enumerate(values)])),round_sd)
            display(Latex(r'$ \sigma =  %s $' % sd))
            print('****************************')
            return sd

    class ZValues:
        def __init__(self):
            pass

        class Normal:
            def __init__ (self):
                pass

            class Population:
                def __init__(self):
                    pass

                def get_z_value(x, mu, sigma, round_z=2):
                    z = round((x - mu) / sigma, round_z)
                    display(Latex(r'$ z = \frac{x - \mu}{\sigma} = \frac{%s - %s}{%s} = %s $' % (x, mu, sigma, z)))
                    return z

                def get_x_value(z, mu, sigma, round_z=2):
                    x = round((z * sigma) + mu, round_z)
                    display(Latex(r'$ x = z \sigma + \mu = %s * %s + %s = %s $' % (z, sigma, mu, x)))
                    return z

            class Sample:
                def __init__(self):
                    pass

                def get_mu(mu):
                    display(Latex(r'$ \mu_{\bar{x}} = \mu = %s $' % mu))
                    return mu

                def get_sigma(n, sigma,round_sigma=4):
                    _sigma = round(sigma / math.sqrt(n), round_sigma)
                    display(Latex(
                        r'$ \sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}} = \frac{%s}{\sqrt{%s}} = %s $' % (sigma, n, _sigma)))
                    return _sigma

                def get_z_value(x, n, mu, sigma, round_z=2):
                    _sigma = MyStats.ZValues.Normal.Sample.get_sigma(n, sigma)
                    _mu = MyStats.ZValues.Normal.Sample.get_mu(mu)
                    z = round((x - _mu) / _sigma, round_z)
                    display(Latex(r'$ z = \frac{\bar{x} - \mu}{\frac{\sigma}{\sqrt{n}}} = \frac{%s - %s}{%s} = %s $' % (
                    x, _mu, _sigma, z)))
                    return z

        class Binomial:
            def __init__(self):
                pass

            class Population:
                def __init__(self):
                    pass

                def get_mu(n, p, round_mu=3):
                    mu = round(n * p, round_mu)
                    display(Latex(r'$ \mu = np = %s * %s = %s $' % (n, p, mu)))
                    return mu

                def get_sigma(n, p, q, round_sigma=2):
                    sigma = round(math.sqrt(n * p * q), round_sigma)
                    display(Latex(r'$ \sigma = \sqrt{npq} = \sqrt{%s*%s*%s} = %s $' % (n, p, q, sigma)))
                    return sigma

                def get_z_value(x, n, p, q, round_z=2, round_mu=3, round_sigma=2):
                    sigma = MyStats.ZValues.Binomial.Population.get_sigma(n, p, q,round_sigma )
                    mu = MyStats.ZValues.Binomial.Population.get_mu(n, p, round_mu)
                    z = round((x - mu) / sigma, round_z)
                    display(Latex(r'$ z = \frac{x - \mu}{\sigma} = \frac{%s - %s}{%s} = %s $' % (
                        x, mu, sigma, z)))
                    return z

            class Sample:
                def __init__(self):
                    pass

                def get_mu(p):
                    mu = p
                    display(Latex(r'$ \mu_{\hat{p}} = p = %s $' % (p)))
                    return mu

                def get_sigma(n, p, q, round_sigma=3):
                    sigma = round(math.sqrt((p*q)/n), round_sigma)
                    display(Latex(r'$ \sigma_{\hat{p}} = \sqrt{\frac{pq}{n}} = \sqrt{\frac{%s * %s}{%s}} = %s $' % (p, q, n, sigma)))
                    return sigma

                def get_z_value(x, n, p, q, round_z=2, round_sigma=3):
                    sigma = MyStats.ZValues.Binomial.Sample.get_sigma(n, p, q, round_sigma)
                    mu = MyStats.ZValues.Binomial.Sample.get_mu(p)
                    z = round((x - mu) / sigma, round_z)
                    display(Latex(r'$ z = \frac{x - \mu}{\sigma} = \frac{%s - %s}{%s} = %s $' % (
                        x, mu, sigma, z)))
                    return z

        class Proportion:
            def __init__(self):
                pass

            def get_mu(p):
                display(Latex(r'$ \mu_{\hat{p}} = %s $' % p))
                return p

            def get_sigma(p,n):
                q = round(1-p, 2)
                sigma = round(math.sqrt((p*q)/n),3)
                display(Latex(r'$ \sigma_{\hat{p}} = \sqrt{\frac{pq}{n}} = \sqrt{\frac{%s * %s}{%s}}  = %s$' %(p,q,n, sigma)))
                return sigma


            def get_z_value(phat, n, p, round_q=2, round_z=2):
                q = round(1-p, round_q)
                z = round((phat - p)/math.sqrt((p*q)/n), round_z)
                display(Latex(r'$ z = \frac{\hat{p} - p}{ \sqrt{\frac{pq}{n}}}  = \frac{%s - %s}{ \sqrt{\frac{%s * %s}{%s}}} = %s $' % (phat, p, p, q, n,z)))
                return z

    class TValues:
        def __init__(self):
            pass

        class HypothesisTesting:
            def __init__(self):
                pass

            def get_t_value(xbar, mu, s, n, round_t=2):
                print("REMEMBER TO USE VALUE FROM THE TABLE FOR PROBABILITY")
                t = round((xbar - mu) / (s/math.sqrt(n)), round_t)
                display(Latex(r'$ t = \frac{\bar{x} - \mu}{ \frac{s}{\sqrt{n}}} = \frac{%s - %s}{\frac{%s}{\sqrt{%s}}} = %s $' % (xbar, mu, s, n, t)))
                return t

        class PairedDifferences:
            def __init__(self):
                pass

            def get_t_value(dbar, ssubd, n, mu, round_t=3):
                print("REMEMBER TO USE VALUE FROM THE TABLE FOR PROBABILITY")
                if(mu == 0):
                    tvalue = round((dbar * math.sqrt(n)) / ssubd, round_t)
                    display(Latex(r'$ t = \frac{  \bar{d} - 0  }{\frac{ s_{d}  }{ \sqrt{n} }} = \frac{\bar{d} * \sqrt{n}}{s_{d}} = \frac{%s * %s}{%s} = %s $' %(dbar, n,ssubd, tvalue)))
                else:
                    tvalue = round((dbar - mu ) / (ssubd / math.sqrt(n)), round_t)
                    display(Latex(
                        r'$ t = \frac{  \bar{d} - \mu_{d}  }{\frac{ s_{d}  }{ \sqrt{n} }} = \frac{%s - %s}{\frac{%s}{\sqrt{%s}}} = %s $' % (
                        dbar, mu, ssubd, n, tvalue)))

                return tvalue

    class CriticalValues:
        def __init__(self):
            pass
        
        def get_normal_distribution_cv(level_of_confidence, round_cv=3):
            critical_value = round(norm.ppf(1-((1-level_of_confidence)/2)),round_cv)
            return critical_value
        
        def get_students_distribution_cv(level_of_confidence, n, round_cv=3):
            df = n-1
            if(df > 30):
                print("CHECK TABLE, NEED TO ADJUST DF FOR HOMEWORK : d.f. = %s" %df)
            adjusted_level_of_confidence = level_of_confidence + (1-level_of_confidence)/2
            return(round(t.ppf(adjusted_level_of_confidence, df),round_cv))

    class ConfidenceIntervals:
        def __init__(self):
            pass

        def get_ci_for_binomial(n, r, level_of_confidence, round_phat=4, round_qhat=3, round_np=2, round_nq=2, round_E =4, round_left=2, round_right=2):
            
            test_against_5 = MyStats.Utils.test_against_5
            
            phat = round(r/n, round_phat)
            qhat = round(1-phat, round_qhat)
            np = round(phat*n, round_np)
            nq = round(qhat*n, round_nq)

            critical_value = MyStats.CriticalValues.get_normal_distribution_cv(level_of_confidence)

            E = round(critical_value*(math.sqrt((phat*qhat)/n)),round_E)

            left = round(phat-E,round_left)
            right = round(phat+E,round_right)

            display(Latex(r'$\hat{p} = \frac{r}{n} = \frac{%s}{%s} = %s $' %(r,n,phat)))
            display(Latex(r'$\hat{q} = 1- \hat{p} = 1-%s = %s$'%(phat,qhat)))

            display(Latex(r'$np \approx n\hat{p} = %s %s 5$ AND $nq \approx n\hat{q} = %s %s 5 $'%(np,test_against_5(np),nq,test_against_5(nq))))
            print(r'np \approx n\hat{p} = %s %s 5 AND nq \approx n\hat{q} = %s %s 5 '%(np,test_against_5(np),nq,test_against_5(nq)))
            display(Latex(r'Since the estimates of np and nq are much greater than 5, it is reasonable to assume np and nq > 5. '
                          r'Then we can use the normal distribution with $\mu = p$ and $\sigma = \sqrt{\frac{pq}{n}}$ to approximate the distribution of $\hat{p}$'))



            display(Latex(r'$E = z_{c} * \sqrt{\frac{\hat{p} * \hat{q}}{n}}  = %s * \sqrt{\frac{(%s) * (%s)}{%s}} = %s$' % (critical_value, phat, qhat, n,E)))

            display(Latex(r'$(\hat{p} - E) < p < (\hat{p} + E) = (%s - %s) < p < (%s + %s) $'%(phat,E,phat,E)))
            display(Latex(r'$ %s < p < %s $'%(left, right)))
            display(Latex(r'We are %s confident that this interval is one of the intervals that contains p' %level_of_confidence))

            display(Latex(r'In repeated sampling, approximately %s of the intervals created from the samples would include p, the THING THaT YOU ARE STUDYING ' % level_of_confidence))
            display(Latex(r'If many additional samples of size n = %s were drawn from this population and a confidence interval were created from each such sample, approximately %s of those confidence intervals would contain p ' % (n,level_of_confidence)))

        def get_ci_with_sigma_known(level_of_confidence, n, sigma, xbar, round_E=3, round_left=3, round_right=3):
            MyStats.ConfidenceIntervals.normal_check(n)
            critical_value = MyStats.CriticalValues.get_normal_distribution_cv(level_of_confidence)
            E = round(critical_value *(sigma/math.sqrt(n)), round_E)
            leftside = round(xbar - E, round_left)
            rightside = round(xbar + E, round_right)

            display(Latex(r'$\bar{x} - E \leq \mu \leq \bar{x} + E$'))
            display(Latex(r'$E = z_{c}\frac{\sigma}{\sqrt{n}} = %s\frac{%s}{\sqrt{%s}} = %s$'%(critical_value, sigma, n, E)))
            display(Latex(r'$%s- %s \leq \mu \leq %s + %s$' %(xbar,E,xbar,E)))
            display(Latex(r'$%s \leq \mu \leq %s$' %(leftside,rightside)))
            display(Latex(r'We conclude with %s confidence that the interval from %s to %s is one that contains the population mean of $\mu$. '%(level_of_confidence,leftside,rightside)))

        def get_ci_with_sigma_unknown(level_of_confidence,n,s,xbar, critical_value=0, round_E=3, round_left=3, round_right=3):
            MyStats.ConfidenceIntervals.normal_check(n)
            if critical_value == 0:
                critical_value = MyStats.CriticalValues.get_students_distribution_cv(level_of_confidence,n)
            E = round(critical_value *(s/math.sqrt(n)), round_E)
            leftside = round(xbar - E, round_left)
            rightside = round(xbar + E, round_right)
            df = n-1

            display(Latex(r'$df = %s $' % df))
            display(Latex(r'$t_{c} = %s $' % critical_value))
            display(Latex(r'$\bar{x} - E < \mu < \bar{x} + E$'))
            display(Latex(r'$E = t_{c}\frac{s}{\sqrt{n}} = %s\frac{%s}{\sqrt{%s}} = %s$'%(critical_value, s, n, E) ))
            display(Latex(r'$%s- %s < \mu < %s + %s$' %(xbar,E,xbar,E)))
            display(Latex(r'$%s < \mu < %s$' %(leftside,rightside)))
            display(Latex(r'We conclude with %s confidence that the interval from %s to %s is one that contains the population mean of $\mu$. '%(level_of_confidence,leftside,rightside)))

        def get_ci_paired_samples_with_sigma_known(level_of_confidence, x1, x2, n1, n2, sigma1, sigma2, round_left=3, round_right=3, round_E=2):

            critical_value = MyStats.CriticalValues.get_normal_distribution_cv(level_of_confidence)
            print(critical_value)
            E = round(critical_value * math.sqrt( ((sigma1**2) / n1) + ((sigma2**2) / n2) ), round_E)
            print(E)

            leftside = round(x1-x2 - E, round_left)
            rightside = round(x1-x2 + E, round_right)

            display(Latex(r'$E = z_{c} * \sqrt{  \frac{ \sigma_{1}^2 }{n_{1}} + \frac{ \sigma_{2}^2 }{n_{2}} } =  '
                          r'%s * \sqrt{  \frac{ %s^2 }{%s} + \frac{ %s^2 }{%s} } = %s  $' % (critical_value,sigma1, n1, sigma2, n2, E)))
            display(Latex(r'$( \bar{x_{1}} - \bar{x_{2}}) - E < \mu_{1} - \mu_{2} < ( \bar{x_{1}} - \bar{x_{2}}) + E$'))
            display(Latex(r'$( %s - %s) - %s < \mu_{1} - \mu_{2} < ( %s -%s) + %s$' % (x1,n1,E,x2,n2,E)))
            display(Latex(r'$%s< \mu_{1} - \mu_{2} < %s$' % (leftside, rightside)))
            display(Latex(r'We conclude with %s confidence that the interval from %s to %s is one that contains '
                          r'the population difference $\mu_{1} - \mu_{2}$ where $\mu_{1}$ represents XXX and $\mu_{2}$ represents YYY . '%(level_of_confidence,leftside,rightside)))

            return E

        def get_ci_paired_samples_with_sigma_unknown(level_of_confidence, x1, x2, n1, n2, s1, s2,round_left=3, round_right=3, round_E=2):

            n_for_students = n1 if n1 < n2 else n2
            critical_value = MyStats.CriticalValues.get_students_distribution_cv(level_of_confidence, n_for_students)
            print(critical_value)
            E = round(critical_value * math.sqrt(((s1 ** 2) / n1) + ((s2 ** 2) / n2)), round_E)
            print(E)

            leftside = round(x1 - x2 - E, round_left)
            rightside = round(x1 - x2 + E, round_right)

            display(Latex(r'$E = z_{c} * \sqrt{  \frac{ s_{1}^2 }{n_{1}} + \frac{ s_{2}^2 }{n_{2}} } =  '
                          r'%s * \sqrt{  \frac{ %s^2 }{%s} + \frac{ %s^2 }{%s} } = %s  $' % (
                          critical_value, s1, n1, s2, n2, E)))
            display(Latex(r'$( \bar{x_{1}} - \bar{x_{2}}) - E < \mu_{1} - \mu_{2} < ( \bar{x_{1}} - \bar{x_{2}}) + E$'))
            display(Latex(r'$( %s - %s) - %s < \mu_{1} - \mu_{2} < ( %s -%s) + %s$' % (x1, n1, E, x2, n2, E)))
            display(Latex(r'$%s< \mu_{1} - \mu_{2} < %s$' % (leftside, rightside)))
            display(Latex(r'We conclude with %s confidence that the interval from %s to %s is one that contains '
                          r'the population difference $\mu_{1} - \mu_{2}$ where $\mu_{1}$ represents XXX and $\mu_{2}$ represents YYY . ' % (
                          level_of_confidence, leftside, rightside)))

            return E

        def get_ci_paired_samples_porportions(level_of_confidence, n1, n2, r1, r2, round_left=3, round_right=3, round_E=3, round_hats=4):
            critical_value = MyStats.CriticalValues.get_normal_distribution_cv(level_of_confidence)
            print(critical_value)

            phat1 = round(r1/n1,round_hats)
            phat2 = round(r2/n2, round_hats)

            qhat1 = round(1 - phat1, round_hats)
            qhat2 = round(1 - phat2, round_hats)

            MyStats.ConfidenceIntervals.normal_check_proportions(n1, phat1, qhat1, n2, phat2, qhat2)
            sigmahat = round(math.sqrt(((phat1 * qhat1) / n1) + ((phat2 * qhat2) / n2)), 4)
            E = round(critical_value * sigmahat, round_E)
            print(E)

            leftside = round(phat1-phat2 - E, round_left)
            rightside = round(phat1-phat2 + E, round_right)




            display(Latex(r'$E '
                          r'= z_{c} * \hat{\sigma} '
                          r'=  z_{c} * \sqrt{  \frac{  \hat{p_{1}} * \hat{q_{1}}  }{n_{1}} + \frac{ \hat{p_{2}} * \hat{q_{2}}  }{n_{2}} } '
                          r'=  %s * \sqrt{  \frac{ %s*%s }{%s} + \frac{ %s*%s }{%s} } '
                          r'=  %s * %s '
                          r'= %s  $' % (critical_value,phat1, qhat1, n1, phat2, qhat2, n2,critical_value, sigmahat, E)))


            display(Latex(r'$( \hat{p_{1}} - \hat{p_{2}}) - E < p_{1} - p_{2} < ( \hat{p_{1}} - \hat{p_{2}}) + E$'))
            display(Latex(r'$( %s - %s) - %s < \mu_{1} - \mu_{2} < ( %s -%s) + %s$' % (phat1,phat2,E,phat1,phat2,E)))
            display(Latex(r'$%s< \mu_{1} - \mu_{2} < %s$' % (leftside, rightside)))

            if(leftside < 0 and rightside < 0):
                display(Latex(r'The %s  confidence interval contains only negative values. '
                              r'In this case, we conclude that $\mu_{1} - \mu_{2} < 0 $ or $ p_{1} - p_{2} < 0$'
                              r'and we are therefore %s  confident that $\mu_{1} < \mu_{2}$ or $p_{1} < p_{2}$ ' % (level_of_confidence, level_of_confidence)))
            elif(leftside > 0 and rightside > 0):
                display(Latex(r'The %s  confidence interval contains only positive values. '
                              r'In this case, we conclude that $\mu_{1} - \mu_{2} > 0 $ or $ p_{1} - p_{2} > 0$'
                              r'and we are therefore %s  confident that $\mu_{1} > \mu_{2}$ or $p_{1} > p_{2}$ ' % (level_of_confidence, level_of_confidence)))
            else:
                display(Latex(r'The %s  confidence interval contains both positive and negative values. '
                              r'In this case, we cannot conclude that either $\mu_{1}$ or $\mu_{2}$ or $ p_{1}$ or $p_{2}$'
                              r'is larger ' % (level_of_confidence)))



        def normal_check_proportions(n1, phat1, qhat1, n2, phat2, qhat2):
            n1phat1 = round(n1 * phat1,2)
            n1qhat1 = round(n1 * qhat1,2)

            n2phat2 = round(n2 * phat2,2)
            n2qhat2 = round(n2 * qhat2,2)

            if (n1phat1 > 5 and n2phat2 > 5 and n1qhat1 > 5 and n2qhat2 > 5):
                display(Latex(r'$n_{1}*\hat{p_{1}} = %s $'
                              r' and $n_{1}*\hat{q_{1}} = %s$'
                              r' and $n_{2}*\hat{p_{2}} = %s$'
                              r' and $n_{2}*\hat{q_{2}} = %s$' % (n1phat1, n1qhat1, n2phat2, n2qhat2)))
                print("Are all greater than 5")

            else:
                print("The n's are not large enough....need some words here...")

        def normal_check(n):


            if (n  >= 30):
                display(Latex(r'$ n  = %s \geq  30$' % n))
                display(Latex(
                    r' The requirement that the x distribution be approximately normal can be dropped since the sample size is large enough to ensure that the $\bar{x}$ distribution is approximately normal.'))
            else:
                display(Latex(r'$ n = %s < 30 $' % n))
                a = input("Does x have a mound shaped distribution?")
                if a == 'y':
                    display(Latex(
                        r' The x distribution is normal, so the $\bar{x}$ distribution is also normal and $\sigma$ is known. '))
                    return

    class SampleSizes:
        def __init__(self):
            pass


        class Binomial:
            def __init__(self):
                pass 
            def with_p_estimate(p,level_of_confidence, E, round_n=2, round_z=2):
                critical_value = round(MyStats.CriticalValues.get_normal_distribution_cv(level_of_confidence), round_z)
                n = round(p*(1-p)*((critical_value/E)**2),round_n)

                n_rounded = math.ceil(n)
                display(Latex(r'$ n = p(1-p)(\frac{z_{c}}{E})^2 = %s(1-%s)(\frac{%s}{%s})^2 = %s = %s $' %(p,p,critical_value, E, n, n_rounded)))
                return n_rounded
                
            def without_p_estimate(level_of_confidence, E, round_n=2, round_z=2):
                critical_value = round(MyStats.CriticalValues.get_normal_distribution_cv(level_of_confidence),round_z)

                n = round(1/4*((critical_value/E)**2),round_n)
                n_rounded = math.ceil(n)
                display(Latex(r'$ n = 1/4(\frac{z_{c}}{E})^2 = 1/4(\frac{%s}{%s})^2 = %s = %s $' %(critical_value, E, n, n_rounded)))
                
                return n_rounded

            def min_value_of_n_for_p(p, expected_value):
                display(Latex(r'$ %s \leq \mu = np $' % expected_value))
                display(Latex(r'$ \frac{%s}{%s} \leq n $' % (expected_value, p)))

                n = round(expected_value/p,3)

                display(Latex(r'$ \frac{%s}{%s} \leq %s $' % (expected_value, p, n)))

                return n

        class Normal:
            def __init__(self):
                pass 
            
            def with_sigma_known(level_of_confidence, E, sigma, round_n=3):
                critical_value = MyStats.CriticalValues.get_normal_distribution_cv(level_of_confidence)

                display(Latex(r'$ n= (\frac{z_{c}\sigma}{E})^2 $'))

                n=round(((critical_value * sigma)/E)**2, round_n)
                n_rounded=math.ceil(((critical_value * sigma)/E)**2)

                display(Latex(r'$ n= (\frac{(%s)(%s)}{%s})^2  = %s $'%(critical_value,sigma,E, n)))
                display(Latex('Rounded up to next whole number : %s' % n_rounded))

    class Probability:
        def __init__(self):
            pass

        class Normal:
            def __init__ (self):
                pass

            class Population:
                def __init__(self):
                    pass

                def get_lte_x(x, mu, sigma, round_p=4):
                    display(Latex(r'$ P(x \leq %s) $' % x))
                    z = MyStats.ZValues.Normal.Population.get_z_value(x, mu, sigma)
                    p = round(norm.cdf(z), round_p)
                    display(Latex(r'$ P(x \leq %s) = %s $' % (x, p)))
                    return {"p":p, "z":z}

                def get_gte_x(x, mu, sigma, round_ltp=4, round_p=4):
                    display(Latex(r'$ P(x \geq %s) $' % x))
                    z = MyStats.ZValues.Normal.Population.get_z_value(x, mu, sigma)
                    ltp = round(norm.cdf(z), round_ltp)
                    p = round(1-norm.cdf(z), round_p)
                    display(Latex(r'$ P(x \geq %s) = P(z \geq %s) = 1 - P(z < %s) = 1-%s = %s $' % (x, z, z, ltp, p)))
                    return {"p": p, "z": z}

                def get_between_x1_x2(x1, x2, mu, sigma, round_p1=4, round_p2=4):
                    display(Latex(r'$ P(%s \leq x \leq %s) $' % (x1, x2)))
                    z1 = MyStats.ZValues.Normal.Population.get_z_value(x1, mu, sigma)
                    z2 = MyStats.ZValues.Normal.Population.get_z_value(x2, mu, sigma)

                    p1 = round(norm.cdf(z1), round_p1)
                    p2 = round(norm.cdf(z2), round_p2)

                    p = round(p2-p1, 4)

                    display(Latex(r'$ P(%s \leq x \leq %s) = P(%s \leq z \leq %s) = P(z \leq %s) - P(z \leq %s) = %s - %s = %s $' % (x1, x2, z1, z2, z2, z1, p2,p1,p )))
                    return {"p":p, "z1":z1, "z2":z2}

                class ZFromP:
                    def __init__(self):
                        pass

                    def left(p, round_p=2):
                        return round(norm.ppf(p),round_p)

                    def right(p, round_p=2):
                        return round(norm.ppf(1-p),round_p)

                    def between(p, round_p=2):
                        z = round(norm.ppf((1-p)/2),round_p)
                        return (z, abs(z))

            # Also known as the sample distribution and the Central Limit Theorem....
            class Xbar:
                def __init__(self):
                    pass

                def get_lte_x(x, n,  mu, sigma, round_p=4):
                    display(Latex(r'$ P(x \leq %s) $' % x))
                    z = MyStats.ZValues.Normal.Sample.get_z_value(x, n, mu, sigma)
                    p = round(norm.cdf(z), round_p)
                    display(Latex(r'$ P(x \leq %s) = P(z \leq %s) = %s $' % (x, z, p)))
                    return {"p":p , "z":z}

                def get_gte_x(x, n,  mu, sigma, round_ltp=4, round_p=4):
                    display(Latex(r'$ P(x \geq %s) $' % x))
                    z = MyStats.ZValues.Normal.Sample.get_z_value(x, n, mu, sigma)
                    if(z < 0):
                        print('z is less than zero in a greater than calc. Weird. Using Absolute value...')
                        z = abs(z)
                    ltp = round(norm.cdf(z), round_ltp)
                    p = round(1 - norm.cdf(z), round_p)
                    display(Latex(r'$ P(x \geq %s) = P(z \geq %s) = 1 - P(z < %s) = 1-%s = %s $' % (x, z, z, ltp, p)))
                    return {'p' : p, 'z' : z}

                def get_between_x1_x2(x1, x2, n, mu, sigma, round_p1=4, round_p2=4, round_p=4):
                    display(Latex(r'$ P(%s \leq x \leq %s) $' % (x1, x2)))
                    z1 = MyStats.ZValues.Normal.Sample.get_z_value(x1, n, mu, sigma)
                    z2 = MyStats.ZValues.Normal.Sample.get_z_value(x2, n, mu, sigma)

                    p1 = round(norm.cdf(z1), round_p1)
                    p2 = round(norm.cdf(z2), round_p2)

                    p = round(p2 - p1, round_p)

                    display(Latex(
                        r'$ P(%s \leq x \leq %s) = P(%s \leq z \leq %s) = P(z \leq %s) - P(z \leq %s) = %s - %s = %s $' % (
                        x1, x2, z1, z2, z2, z1, p2, p1, p)))
                    return {'p':p, 'z1':z1, 'z2':z2}

                class ZFromP:
                    def __init__(self):
                        pass

                    def left(p, round_p=2):
                        return round(norm.ppf(p), round_p)

                    def right(p, round_p=2):
                        return round(norm.ppf(1 - p), round_p)

                    def between(p, round_p=2):
                        z = round(norm.ppf((1 - p) / 2), round_p)
                        return (z, abs(z))

        class Binomial:
            def __init__(self):
                pass

            class Population:
                def __init__(self):
                    pass

                def get_x(x, n, p, round_q=3, round_p=4):

                    MyStats.Probability.Binomial.Population.normal_check(n, p)
                    q = round(1 - p, round_q)
                    display(Latex(r'$ P(r = %s) = P(x = %s)   $' % (x, x)))
                    z = MyStats.ZValues.Binomial.Population.get_z_value(x, n, p, q)

                    p = round(norm.pdf(z), round_p)
                    display(Latex(r'$ P(x = %s) = %s $' % (x, p)))
                    return {"p": p, "z": z}

                def get_lte_x(x, n, p, round_q=3, round_p=4):

                    MyStats.Probability.Binomial.Population.normal_check(n,p)
                    q = round(1-p,round_q)
                    corrected_x = x + .5
                    display(Latex(r'$ P(r \leq %s) = P(x \leq %s)   $' % (x, corrected_x)))
                    z = MyStats.ZValues.Binomial.Population.get_z_value(corrected_x, n, p, q)

                    p = round(norm.cdf(z), round_p)
                    display(Latex(r'$ P(x \leq %s) = %s $' % (corrected_x, p)))
                    return {"p" : p,"z": z}

                def get_gte_x(x, n, p, round_q=3, round_ltp=4, round_p=4):
                    MyStats.Probability.Binomial.Population.normal_check(n, p)
                    q = round(1 - p,round_q)
                    corrected_x = x-.5
                    display(Latex(r'$ P(r \geq %s)  = P(x \geq %s) $' % (x,corrected_x) ))
                    z = MyStats.ZValues.Binomial.Population.get_z_value(corrected_x, n, p, q)
                    ltp = round(norm.cdf(z), round_ltp)
                    p = round(1 - norm.cdf(z), round_p)
                    display(
                        Latex(r'$ P(x \geq %s) = P(z \geq %s) = 1 - P(z < %s) = 1-%s = %s $' % (corrected_x, z, z, ltp, p)))
                    return {'p':p ,'z':z}

                def get_between_x1_x2(x1, x2, n, p, round_q=3, round_p1=4, round_p2=4, round_p=4):
                    MyStats.Probability.Binomial.Population.normal_check(n, p)
                    q = round(1 - p,round_q)
                    corrected_x1 = x1-.5
                    corrected_x2 = x2+.5
                    display(Latex(r'$ P(%s \leq r \leq %s)  = P(%s \leq x \leq %s) $' % (x1, x2, corrected_x1, corrected_x2)))
                    z1 = MyStats.ZValues.Binomial.Population.get_z_value(corrected_x1, n, p, q)
                    z2 = MyStats.ZValues.Binomial.Population.get_z_value(corrected_x2, n, p, q)

                    p1 = round(norm.cdf(z1), round_p1)
                    p2 = round(norm.cdf(z2), round_p2)

                    p = round(p2 - p1, round_p)

                    display(Latex(
                        r'$ P(%s \leq x \leq %s) = P(%s \leq z \leq %s) = P(z \leq %s) - P(z \leq %s) = %s - %s = %s $' % (
                            corrected_x1, corrected_x2, z1, z2, z2, z1, p2, p1, p)))
                    return (p , z1, z2)



                def normal_check(n, p, round_q=3, round_np=3, round_nq=3):
                    q = round(1 - p,round_q)
                    np = round(n*p,round_np)
                    nq = round(n*q,round_nq)
                    if (n * p > 5):
                        display(Latex(r'$ np  = %s > 5 $' % np))
                    else:
                        display(Latex(r'$ np = %s \leq 5 $' % np))

                    if (n * q > 5):
                        display(Latex(r'$ nq = %s > 5$' % nq))
                    else:
                        display(Latex(r'$ nq = %s \leq 5 $' % nq))

            class Sample:
                def __init__(self):
                    pass

                def get_lte_x(x, n, p, round_p=4):

                    MyStats.GetProbability.Binomial.Sample.normal_check(n, p)

                    q = 1-p
                    corrected_x = x + .5
                    display(Latex(r'$ P(r \leq %s) = P(x \leq %s)   $' % (x, corrected_x)))
                    z = MyStats.GetProbability.Binomial.Sample.GetZValue(corrected_x, n, p, q)

                    p = round(norm.cdf(z), round_p)
                    display(Latex(r'$ P(x \leq %s) = %s $' % (corrected_x, p)))
                    return (p, z)

                def get_gte_x(x, n, p, round_ltp=4, round_p=4):

                    MyStats.GetProbability.Binomial.Sample.normal_check(n, p)
                    q = 1 - p
                    corrected_x = x-.5
                    display(Latex(r'$ P(r \geq %s)  = P(x \geq %s) $' % (x,corrected_x) ))
                    z = MyStats.GetProbability.Binomial.Sample.GetZValue(corrected_x, n, p, q)
                    ltp = round(norm.cdf(z), round_ltp)
                    p = round(1 - norm.cdf(z), round_p)
                    display(
                        Latex(r'$ P(x \geq %s) = P(z \geq %s) = 1 - P(z < %s) = 1-%s = %s $' % (corrected_x, z, z, ltp, p)))
                    return (p, z)

                def get_between_x1_x2(x1, x2, n, p, round_p1=4, round_p2=4, round_p=4):
                    self.normal_check(n, p)
                    q = 1 - p
                    corrected_x1 = x1-.5
                    corrected_x2 = x2+.5
                    display(Latex(r'$ P(%s \leq r \leq %s)  = P(%s \leq x \leq %s) $' % (x1, x2, corrected_x1, corrected_x2)))
                    z1 = MyStats.GetProbability.Binomial.Sample.GetZValue(corrected_x1, n, p, q)
                    z2 = MyStats.GetProbability.Binomial.Sample.GetZValue(corrected_x2, n, p, q)

                    p1 = round(norm.cdf(z1), round_p1)
                    p2 = round(norm.cdf(z2), round_p1)

                    p = round(p2 - p1, round_p)

                    display(Latex(
                        r'$ P(%s \leq x \leq %s) = P(%s \leq z \leq %s) = P(z \leq %s) - P(z \leq %s) = %s - %s = %s $' % (
                            corrected_x1, corrected_x2, z1, z2, z2, z1, p2, p1, p)))
                    return (p, z1, z2)

                def normal_check(n, p):
                    q = 1-p
                    if(n*p > 5):
                        display(Latex(r'$ np  = %s > 5 $' % (n*p)))
                    else:
                        display(Latex(r'$ np = %s \leq 5 $' % (n*p)))

                    if(n*q > 5):
                        display(Latex(r'$ nq = %s > 5$' % (n*q)))
                    else:
                        display(Latex(r'$ nq = %s \leq 5 $' % (n*q)))

        class Experiment:
            def __init__(self):
                pass

            def get_r(n,p,r):
                # return binom.cdf(r, n, p)
                q = round(1-p,3)
                display(Latex(r'$ P(r) = \frac{n!}{r!(n-r)!}  p^r  q^{n-r}$'))
                display(Latex(r'$ P(r) = \frac{%s!}{%s!(%s-%s)!}  %s^%s  %s^{%s-%s}$' %(n, r, n, r, p, r, q, n, r)))
                calc = (math.factorial(n)/( (math.factorial(r) * math.factorial(n-r)  ) )) * (p**r) * (q**(n-r))
                print ('by the formula : %s' %round(calc,3))
                return round(binom.pmf(r, n, p), 3)

            def get_lte_r(n, p, r):
                return round(binom.cdf(r, n, p),3)

            def get_gte_r(n, p, r):
                rounded = 0
                for x in range(r,n+1):
                    rounded += round(binom.pmf(x, n, p), 3)
                    print('%s : %s' %(x,round(binom.pmf(x, n, p), 3)))
                print(rounded)
                return round(1- binom.cdf(r-1, n, p),3)

        class Students:
            def __init__(self):
                pass

            def get_lte_x(xbar, n, s, mu, round_p=4):

                df = n-1
                display(Latex(r'$ df =  %s $' % df))

                display(Latex(r'$ P(\bar{x} \leq %s) $' % xbar))

                tvalue = MyStats.TValues.HypothesisTesting.get_t_value(xbar, mu, s, n)
                p = round(t.cdf(tvalue, df), round_p)
                display(Latex(r'$ P(\bar{x} \leq %s) = P(t \leq %s) = %s $' % (xbar, tvalue, p)))
                return {'p': p, 't': tvalue}

            def get_gte_x(xbar, n, s, mu, round_ltp=4, round_p=4):

                df = n - 1
                display(Latex(r'$ df =  %s $' % df))

                display(Latex(r'$ P(\bar{x} \geq %s) $' % xbar))
                tvalue = MyStats.TValues.HypothesisTesting.get_t_value(xbar, mu, s, n)
                if (tvalue < 0):
                    print('tvalue is less than zero in a greater than calc. Weird. Using Absolute value...')
                    tvalue = abs(tvalue)

                p = round(1 - t.cdf(tvalue, df), round_p)
                ltp = round(t.cdf(tvalue, df), round_ltp)

                display(Latex(r'$ P(\bar{x} \geq %s) = P(t \geq %s) = 1 - P(t < %s) = 1-%s = %s $' % (xbar, tvalue, tvalue, ltp, p)))
                return {'p': p, 't': tvalue}

            def get_between_x1_x2(xbar1, xbar2, n, s, mu, round_p1=4, round_p2=4, round_p=4):
                df = n - 1
                display(Latex(r'$ df =  %s $' % df))

                display(Latex(r'$ P(%s \leq x \leq %s) $' % (x1, x2)))
                tvalue1 = MyStats.TValues.HypothesisTesting.get_t_value(xbar1, mu, s, n)
                tvalue2 = MyStats.TValues.HypothesisTesting.get_t_value(xbar2, mu, s, n)

                p1 = round(t.cdf(tvalue1, df), round_p1)
                p2 = round(t.cdf(tvalue2, df), round_p2)

                p = round(p2 - p1, round_p)

                display(Latex(
                    r'$ P(%s \leq \bar{x} \leq %s) = P(%s \leq t \leq %s) = P(t \leq %s) - P(t \leq %s) = %s - %s = %s $' % (
                    xbar1, xbar2, tvalue1, tvalue2, tvalue2, tvalue1, p2, p1, p)))
                return (p, tvalue1, tvalue2)

                return {'p': p, 't1': tvalue1, 't2' : tvalue2}

        class Proportion:
            def __init__ (self):
                pass

            class Sample:
                def __init__(self):
                    pass

                def get_lte_r(r, n, p,  round_p=4, round_phat=4):

                    phat = round(r/n , round_phat)
                    display(Latex(r'$ \hat{p} = \frac{r}{n} = \frac{%s}{%s} = %s $' % (r, n, phat)))
                    display(Latex(r'$ P(\hat{p} \leq %s) $' % phat))

                    z = MyStats.ZValues.Proportion.get_z_value(phat, n, p)
                    p = round(norm.cdf(z), round_p)
                    display(Latex(r'$ P(\hat{p} \leq %s) = P(z \leq %s) = %s $' % (phat, z, p)))
                    return (p, z)

                def get_gte_r(r, n, p, round_phat=4, round_ltp=4, round_p=4):
                    phat = round(r / n, round_phat)
                    display(Latex(r'$ \hat{p} = \frac{r}{n} = \frac{%s}{%s} = %s $' % (r, n, phat)))
                    display(Latex(r'$ P(\hat{p} \geq %s) $' % phat))
                    z = MyStats.ZValues.Proportion.get_z_value(phat, n, p)
                    ltp = round(norm.cdf(z), round_ltp)
                    p = round(1 - norm.cdf(z), round_p)
                    display(Latex(r'$ P(\hat{p} \geq %s) = P(z \geq %s) = 1 - P(z < %s) = 1-%s = %s $' % (phat, z, z, ltp, p)))
                    return (p, z)

                def get_between_r1_r2(r1, r2, n, p, round_phat1=4, round_phat2=4, round_p1=4, round_p2=4, round_p=4):

                    phat1 = round(r1/n,round_phat1)
                    display(Latex(r'$ \hat{p} = \frac{r}{n} = \frac{%s}{%s} = %s $' % (r1, n, phat1)))
                    phat2 = round(r2/n,round_phat2)
                    display(Latex(r'$ \hat{p} = \frac{r}{n} = \frac{%s}{%s} = %s $' % (r2, n, phat2)))

                    display(Latex(r'$ P(%s \leq \hat{p} \leq %s) $' % (phat1, phat2)))
                    z1 = MyStats.ZValues.Proportion.get_z_value(phat1, n, p)
                    z2 = MyStats.ZValues.Proportion.get_z_value(phat2, n, p)

                    p1 = round(norm.cdf(z1), round_p1)
                    p2 = round(norm.cdf(z2), round_p2)

                    p = round(p2 - p1, round_p)

                    display(Latex(
                        r'$ P(%s \leq \hat{p} \leq %s) = P(%s \leq z \leq %s) = P(z \leq %s) - P(z \leq %s) = %s - %s = %s $' % (
                        phat1, phat2, z1, z2, z2, z1, p2, p1, p)))
                    return (p, z1, z2)

                class ZFromP:
                    def __init__(self):
                        pass

                    def left(p, round_p=2):
                        return round(norm.ppf(p), round_p)

                    def right(p, round_p=2):
                        return round(norm.ppf(1 - p), round_p)

                    def between(p, round_p=2):
                        z = round(norm.ppf((1 - p) / 2), round_p)
                        return (z, abs(z))

        class PairedDifferences:
                def __init__(self):
                    pass

                def get_lte_x(dbar, ssubd, n, mu, round_p=4):
                    df = n - 1
                    display(Latex(r'$ df =  %s $' % df))
                    display(Latex(r'$ P(\bar{d} \leq %s) $' % dbar))
                    tvalue = MyStats.TValues.PairedDifferences.get_t_value(dbar, ssubd, n, mu)
                    p = round(t.cdf(tvalue, df), round_p)
                    display(Latex(r'$ P(\bar{d} \leq %s) = P(t \leq %s) = %s $' % (dbar, tvalue, p)))
                    return (p, tvalue)

                def get_gte_x(dbar, ssubd, n, mu, round_ltp=4, round_p=4):
                    df = n - 1
                    display(Latex(r'$ df =  %s $' % df))

                    display(Latex(r'$ P(\bar{d} \geq %s) $' % dbar))
                    tvalue = MyStats.TValues.PairedDifferences.get_t_value(dbar, ssubd, n, mu)

                    p = round(1 - t.cdf(tvalue, df), round_p)
                    ltp = round(t.cdf(tvalue, df), round_ltp)

                    display(Latex(r'$ P(\bar{d} \geq %s) = P(t \geq %s) = 1 - P(t < %s) = 1-%s = %s $' % (
                    dbar, tvalue, tvalue, ltp, p)))
                    return (p, tvalue)

                def get_between_x1_x2(dbar1, dbar2, ssubd, n, mu,  round_p1=4, round_p2=4, round_p=4):
                    df = n - 1
                    display(Latex(r'$ df =  %s $' % df))

                    display(Latex(r'$ P(%s \leq \bar{d} \leq %s) $' % (dbar1, dbar2)))
                    tvalue1 = MyStats.TValues.PairedDifferences.get_t_value(dbar1, ssubd, n, mu)
                    tvalue2 = MyStats.TValues.PairedDifferences.get_t_value(dbar2, ssubd, n, mu)

                    p1 = round(t.cdf(tvalue1, df), round_p1)
                    p2 = round(t.cdf(tvalue2, df), round_p2)

                    p = round(p2 - p1, round_p)

                    display(Latex(
                        r'$ P(%s \leq \bar{d} \leq %s) = P(%s \leq t \leq %s) = P(t \leq %s) - P(t \leq %s) = %s - %s = %s $' % (
                            dbar1, dbar2, tvalue1, tvalue2, tvalue2, tvalue1, p2, p1, p)))
                    return (p, tvalue1, tvalue2)

    class HypothesisTesting:
        def __init__(self):
            pass

        def get_types_of_statistical_tests():
            print('left-tailed - if H1 states that the parameter is less than the value claimed in H0')
            print('right-tailed - if H1 states that the parameter is more than the value claimed in H0')
            print('two-tailed - if H1 states that the parameter is different than the value claimed in H0')

        def get_types_of_errors():
            display(Latex(r'Type 1 : Reject H0 when it is true. Level of significance of a test = $\alpha $' ))
            display(Latex(r'Type 1 : Accept H0 when it is false. Probability of making a type II error = $\beta $ '))
            display(Latex(r'$1 - \beta $ is called the power of a test and represents the probability of rejecting H0 when it is in fact false. '))

        class SigmaIsKnown:
                def __init__(self):
                    pass

                def test_hypothesis(xbar, mu, sigma, n, H0, H1, alpha, test_type):

                    MyStats.HypothesisTesting.SigmaIsKnown.normal_check(n)


                    display(Latex(r'$\alpha =  %s $' % alpha))

                    if(test_type == 'less'):
                        display(Latex(r'$H_{0} : \mu = %s $' %mu))
                        display(Latex(r'$H_{1} : \mu < %s $' % mu))
                        result = MyStats.Probability.Normal.Xbar.get_lte_x(xbar, n, mu, sigma)
                        p = result['p']
                        z = result['z']


                        MyStats.Graphs.Normal.showGraph([z], 'left')

                    elif(test_type == 'greater'):
                        display(Latex(r'$H_{0} : \mu = %s $' %mu))
                        display(Latex(r'$H_{1} : \mu > %s $' % mu))
                        result = MyStats.Probability.Normal.Xbar.get_gte_x(xbar, n, mu, sigma)
                        p = result['p']
                        z = result['z']


                        MyStats.Graphs.Normal.showGraph([z], 'right')

                    elif(test_type == 'different'):
                        display(Latex(r'$H_{0} : \mu = %s $' %mu))
                        display(Latex(r'$H_{1} : \mu \neq %s $' % mu))
                        result = MyStats.Probability.Normal.Xbar.get_gte_x(xbar, n, mu, sigma)
                        print(result['z'])
                        if(result['z'] < 0 ):
                            print("The z value is less than 0, using lte")
                            result = MyStats.Probability.Normal.Xbar.get_lte_x(xbar, n, mu, sigma)
                        p = 2 * result['p']
                        z = result['z']



                        MyStats.Graphs.Normal.showGraph([-z, z], 'two')

                    display(Latex(r'$P-value = %s $' % p))

                    if(p <= alpha):
                        display(Latex(r'We reject the null hypothesis. The data are significant at $\alpha = %s $' % alpha))
                        print('The sample data is sufficient at the %s level to justify rejecting H0. It seems that %s. ' %(alpha, H1))

                    else :
                        print('We fail to reject the null hypothesis')
                        print('There is insufficient evidence at the %s level to reject H0. The data are not statistically significant. It seems %s ' %(alpha, H0))

                def normal_check(n):

                    if (n >= 30):
                        display(Latex(r'$ n  = %s \geq  30$' % n))
                        display(Latex(
                            r' The requirement that the x distribution be approximately normal can be dropped since the'
                            r' sample size is large enough to ensure that the $\bar{x}$ distribution is approximately normal.'
                            r'Since n $\geq$ 30 and we know $\sigma$ we can use the standard distribution.'))
                    else:
                        display(Latex(r'$ n = %s < 30 $' % n))
                        a = input("Does x have a mound shaped distribution?")
                        if a == 'y':
                            display(Latex(
                                r' The x distribution is normal, so the $\bar{x}$ distribution is also normal and $\sigma$ is known. '))
                            return

        class SigmaIsUnKnown:
                def __init__(self):
                    pass

                def test_hypothesis(xbar, mu, n, s, H0, H1, alpha, test_type):

                    MyStats.HypothesisTesting.SigmaIsUnKnown.normal_check(n)


                    display(Latex(r'$\alpha =  %s $' % alpha))

                    if(test_type == 'less'):
                        display(Latex(r'$H_{0} : \mu = %s $' %mu))
                        display(Latex(r'$H_{1} : \mu < %s $' % mu))
                        result = MyStats.Probability.Students.get_lte_x(xbar, n, s, mu)
                        p = result['p']
                        tvalue = result['t']

                        MyStats.Graphs.Normal.showGraph([tvalue], 'left')

                    elif(test_type == 'greater'):
                        display(Latex(r'$H_{0} : \mu = %s $' %mu))
                        display(Latex(r'$H_{1} : \mu > %s $' % mu))
                        result = MyStats.Probability.Students.get_gte_x(xbar, n, s, mu,)
                        p = result['p']
                        tvalue = result['t']


                        MyStats.Graphs.Normal.showGraph([tvalue], 'right')

                    elif(test_type == 'different'):
                        display(Latex(r'$H_{0} : \mu = %s $' % mu))
                        display(Latex(r'$H_{1} : \mu \neq %s $' % mu))
                        result = MyStats.Probability.Students.get_gte_x(xbar, n, s, mu)
                        print("T value : %s" % result['t'])
                        if(result['t'] < 0 ):
                            print("The t value is less than 0, using lte")
                            result = MyStats.Probability.Students.get_lte_x(xbar, n, s, mu)

                        gte_p = result['p']
                        display(Latex(r'P-value for a two tailed test = 2* %s ' % gte_p))
                        p = 2 * result['p']
                        tvalue = result['t']

                        MyStats.Graphs.Normal.showGraph([-tvalue, tvalue], 'two')

                    display(Latex(r'$P-value = %s $' % p))

                    if(p <= alpha):
                        display(Latex(r'$p =  %s \leq \alpha = %s $' % (p,alpha)))
                        display(Latex(r'We reject the null hypothesis. The data are significant at $\alpha = %s $' % alpha))
                        print('The sample data is sufficient at the %s level to justify rejecting H0. It seems that %s. ' %(alpha, H1))

                    else:
                        display(Latex(r'$p =  %s > \alpha = %s $' % (p, alpha)))
                        print('We fail to reject the null hypothesis')
                        print('There is insufficient evidence at the %s level to reject H0. The data are not statistically significant. It seems %s ' %(alpha, H0))

                def normal_check(n):

                    if (n >= 30):
                        display(Latex(r'$ n  = %s \geq  30$' % n))
                        display(Latex(r' Since sample size is large  n $\geq$ 30 and we do not know $\sigma$ we can use the Student\'s '
                            r't  distribution with XX degrees of freedom.'))
                    else:
                        display(Latex(r'$ n = %s < 30 $' % n))
                        a = input("Does x have a mound shaped distribution?")
                        if a == 'y':
                            display(Latex(
                                r'Use a Student\'s distibution The x distribution is mound shaped and symmetric, and $\sigma$ is unknown. '))
                            return

        class Proportion:
                def __init__(self):
                    pass

                def test_hypothesis(r, n, p, H0, H1, alpha, test_type):

                    MyStats.HypothesisTesting.Proportion.normal_check(n,p)


                    display(Latex(r'$\alpha =  %s $' % alpha))

                    if(test_type == 'less'):
                        display(Latex(r'$H_{0} : p = %s $' % p))
                        display(Latex(r'$H_{1} : p < %s $' % p))
                        result = MyStats.Probability.Proportion.Sample.get_lte_r(r, n, p)
                        p = result[0]
                        z = result[1]

                        MyStats.Graphs.Normal.showGraph([z], 'left')

                    elif(test_type == 'greater'):
                        display(Latex(r'$H_{0} : p = %s $' % p))
                        display(Latex(r'$H_{1} : p > %s $' % p))
                        result = MyStats.Probability.Proportion.Sample.get_gte_r(r,n,p)
                        p = result[0]
                        z = result[1]

                        MyStats.Graphs.Normal.showGraph([z], 'right')

                    elif(test_type == 'different'):
                        display(Latex(r'$H_{0} : p = %s $' % p))
                        display(Latex(r'$H_{1} : p \neq %s $' % p))
                        result = MyStats.Probability.Proportion.Sample.get_gte_r(r, n, p)
                        if(result[1] < 0 ):
                            print("The z value is less than 0, using lte")
                            result = MyStats.Probability.Proportion.Sample.get_lte_r(r, n, p)

                        p = 2 * result[0]
                        z = abs(result[1])

                        MyStats.Graphs.Normal.showGraph([-z, z], 'two')

                    display(Latex(r'$P-value = %s $' % p))

                    if(p <= alpha):
                        display(Latex(r'We reject the null hypothesis. The data are significant at $\alpha = %s $' % alpha))
                        print('The sample data is sufficient at the %s level to justify rejecting H0. It seems that %s. ' %(alpha, H1))

                    else :
                        print('We fail to reject the null hypothesis')
                        print('There is insufficient evidence at the %s level to reject H0. The data are not statistically significant. It seems we cannot say  %s ' %(alpha, H1))

                def normal_check(n, p):
                    q = 1-p

                    np = round(n*p,2)
                    nq = round(n*q,2)

                    if ((np > 5) and (nq > 5)):
                        display(Latex(r'Use the standard normal distribution. The $\hat{p}$ distribution is approximates normal when n'
                                      r'is sufficiently large, which it is here because $ np = %s > 5 $ AND $ nq = %s > 5 $'% (np,nq)))
                    else:
                        display(Latex(r'np or nq does not meet minimum requirments' % n))

        class PairedDifference:
                def __init__(self):
                    pass

                def get_stats_from_lists(a, b):

                    # difference of paired samples
                    sample = np.subtract(b, a)
                    print(sample)
                    # standard Deviation
                    sd = round(np.std(sample, ddof=1), 3)
                    # mean of paired differences
                    dbar = round(scipy.mean(sample), 3)
                    # Length of sample data
                    n = len(a)

                    display(Latex(r'$\bar{d} =  %s $' % dbar))
                    display(Latex(r'$s_{d} =  %s $' % sd))

                    return{'ssubd': sd, 'dbar': dbar, 'n': n, 'sample': sample}

                def test_hypothesis(lista, listb, H0, H1, alpha, test_type):

                    list_stats = MyStats.HypothesisTesting.PairedDifference.get_stats_from_lists(lista, listb)
                    dbar = list_stats['dbar']
                    ssubd = list_stats['ssubd']
                    n = list_stats['n']

                    mu = 0

                    MyStats.HypothesisTesting.PairedDifference.normal_check(n)

                    display(Latex(r'$\alpha =  %s $' % alpha))

                    if(test_type == 'less'):
                        display(Latex(r'$H_{0} : \mu_{d} = %s $' % mu))
                        display(Latex(r'$H_{1} : \mu_{d} < %s $' % mu))
                        result = MyStats.Probability.PairedDifferences.get_lte_x(dbar, ssubd, n, mu=0)
                        p = result[0]
                        z = result[1]

                        MyStats.Graphs.Students.showGraph([z], n-1, 'left')

                    elif(test_type == 'greater'):
                        display(Latex(r'$H_{0} : \mu_{d} = %s $' % mu))
                        display(Latex(r'$H_{1} : \mu_{d} > %s $' % mu))
                        result = MyStats.Probability.PairedDifferences.get_gte_x(dbar, ssubd, n, mu=0)
                        p = result[0]
                        z = result[1]

                        MyStats.Graphs.Students.showGraph([z], n-1, 'right')

                    elif(test_type == 'different'):
                        display(Latex(r'$H_{0} : \mu_{d} = %s $' % mu))
                        display(Latex(r'$H_{1} : \mu_{d} \neq %s $' % mu))
                        result = MyStats.Probability.PairedDifferences.get_gte_x(dbar, ssubd, n, mu=0)
                        if(result[1] < 0 ):
                            print("The z value is less than 0, using lte")
                            result = MyStats.Probability.PairedDifferences.get_lte_x(dbar, ssubd, n, mu=0)

                        p = 2 * result[0]
                        z = abs(result[1])

                        MyStats.Graphs.Students.showGraph([-z, z], n-1, 'two')

                    display(Latex(r'$P-value = %s $' % p))

                    if(p <= alpha):
                        display(Latex(r'We reject the null hypothesis. The data are significant at $\alpha = %s $' % alpha))
                        print('At the %s level of significance , the sample mean of differences is sufficiently different from 0 that we conclude the population mean of the differences is not zero.' %(alpha))

                    else :
                        print('We fail to reject the null hypothesis')
                        print('At the %s level of significance, the evidence is insufficient to claim a difference in population mean percentage increase/decrease/difference for %s' %(alpha, H1))

                def normal_check(n):


                    if n > 30 :
                        display(Latex(r'The sample is sufficiently large n = %s > 30. Use A Student\'s t distribution with df = n-1 = %s'% (n, n-1)))
                    else:
                        display(Latex(r'n does not meet minimum requirements. Does d have a normal distribution or has a mound-shaped symmetric distribution?'))

    class Regression:
        def __init__(self):
            pass

        def get_least_squares_line(x,y):

            n = len(x)
            sum_xy = round(MyStats.get_sum_of_product(x, y),2)
            sum_x = round(sum(x),2)
            sum_y = round(sum(y),2)
            sum_x_squared = round(MyStats.get_sum_of_squares(x),2)
            sum_x__squared = round(sum_x**2,2)
            sum_y_squared = round(MyStats.get_sum_of_squares(y),2)
            sum_y__squared = round(sum_y**2,2)
            x_mean = round(MyStats.get_mean(x),3)
            y_mean = round(MyStats.get_mean(y),3)

            b = round((n*sum_xy - sum_x*sum_y) / (n*sum_x_squared-sum_x__squared) ,4)

            top = round(n*sum_xy - sum_x*sum_y,3)
            bottom = round(n*sum_x_squared-sum_x__squared,3)

            display(Latex(r'$ b = \frac{n \sum{xy} - (\sum{x})(\sum{y})}{n * \sum{x^2} - (\sum{x})^2} $'))

            display(Latex(r'$ = \frac{%s * %s - (%s)(%s)}{%s(%s) - %s^2 }'
                      r'$' %(n, sum_xy, sum_x, sum_y, n, sum_x_squared, sum_x)))

            display(Latex(r'$ = \frac{%s}{%s}$' %(top,bottom)))
            display(Latex(r'$ = %s$' % b))


            a = round(y_mean - b*x_mean,3)
            display(Latex(r'$ a = \bar{y} - b*\bar{x} = %s - %s * %s$'%(y_mean, b, x_mean)))
            display(Latex(r'$ a = %s $' %a))

            display(Latex(r'$ \hat{y} = %s + %sx $' % (a, b)))


            return a, b

        def get_y(a, b, x):
            print('a : %s' % a)
            print('b : %s'%b)
            print('x: %s'%x)
            y = round(a + (b * x),2)

            display(Latex(r'$ \hat{y} = b + ax = %s + %s * %s = %s$' % (b,a,x,y)))

            return (y)

        def get_sample_coefficient_r(x,y):

            n = len(x)
            sum_xy = MyStats.get_sum_of_product(x, y)
            sum_x = round(sum(x),2)
            sum_y = round(sum(y),2)
            sum_x_squared = round(MyStats.get_sum_of_squares(x),2)
            sum_x__squared = round(sum_x**2,2)
            sum_y_squared = round(MyStats.get_sum_of_squares(y),2)
            sum_y__squared = round(sum_y**2,2)

            r = round((n*sum_xy - sum_x*sum_y) / ((math.sqrt(n*sum_x_squared - sum_x__squared))*(math.sqrt(n*sum_y_squared - sum_y__squared))),3)

            top = round(n*sum_xy - sum_x*sum_y,2)
            bottomleft = round(math.sqrt(n*sum_x_squared - sum_x__squared),2)
            bottomright = round(math.sqrt(n*sum_y_squared - sum_y__squared),2)


            display(Latex(r'$ r '
                      r'= \frac{n \sum{xy} - (\sum{x})(\sum{y})}{\sqrt{n \sum{x^2} - (\sum{x})^2} * \sqrt{n \sum{y^2} - (\sum{y})^2} }$'))

            display(Latex(r'$ = \frac{%s * %s - (%s)(%s)}{\sqrt{%s(%s) - (%s)^2} * \sqrt{%s(%s) - (%s)^2} }'
                      r'$' %(n, sum_xy, sum_x, sum_y, n, sum_x_squared, sum_x, n, sum_y_squared, sum_y)))

            display(Latex(r'$ = \frac{%s}{(%s)(%s) }$' %(top,bottomleft,bottomright)))
            display(Latex(r'$ = %s$' % (r)))
            return r

        def get_coefficient_of_determination(x, y):
            r =MyStats.Regression.get_sample_coefficient_r(x, y)
            r_squared = round(r**2,2)
            unexplained = round(1-r_squared, 2)
            display(Latex(r'With r = %s : $ r^2= %s$' % (r, r_squared)))


            display(Latex(r'The amount of variation that can be explained by the variation in x and the least squares line is equal ' \
            'to the coefficient of determination $r^2$. With $r = %s$, $r^2 = %s$ so we can explain %s percent of the variation.  This means ' \
            'that %s of the variation is due to random chance or the possibility of lurking variables.' % (r,r_squared, r_squared, unexplained)))


            return r_squared

    class ChiSquare:
        def __init__(self):
            pass

        def get_mode(n):
            if(n>= 3):
                return n-2
        def get_chi_square(arrays, alpha):
            np_arrays = np.array(arrays)
            results = stats.chi2_contingency(np_arrays)

            columns = len(arrays[0])
            rows = len(arrays)

            O_arrays = arrays

            chi_squared = round(results[0],3)
            display(Latex(r'$ \chi^2 = %s$' % chi_squared ))

            probablity = round(results[1],4)
            display(Latex(r'$ p =  %s$' % probablity))

            df = results[2]
            display(Latex(r'$ df =  %s$' % df))

            E_arrays = results[3]
            AllLatex = ''
            all_are_less_than_5 = True
            for i, v in enumerate(E_arrays):
                for i_2, v_2 in enumerate(v):
                    if(v_2 <= 5):
                        all_are_less_than_5 = False
                    AllLatex = AllLatex + r' \frac{(%s -%s)^2}{%s} + ' %(O_arrays[i][i_2], round(v_2, 2),round(v_2, 2))

            display(Latex(r'$ \chi^2 = \sum{\frac{(O-E)^2}{E}} $'))
            display(Latex('$'+AllLatex + ' = %s$'%chi_squared))

            # Assumption check

            if(all_are_less_than_5 == False):
                print("At least one of the expected values was not greater than 5.")
            else:
                print("All expected frequencies are greater than 5. Use the chi-square distribution.")

            #DFs
            print("Since there are %s rows and %s columns, d.f. = (%s - 1)(%s-1) = %s" %(rows, columns, rows, columns, df))

            display(Latex(r'Since d.f. = %s,$\chi^2 = %s $ falls between entries XXX and XXX. '
                          r'Therefore, XXX < P value < XXX' %(df, chi_squared)))

            if(probablity <= alpha):
                display(Latex(r'Since the P value is less than the level of significance $\alpha = %s$, we reject the null hypothesis' % alpha))
                print('At the %s level of significance, there is sufficient evidence to conclude that XXX and XXX are not independent.'%alpha)
            else :
                display(Latex(
                    r'Since the P value is greater than the level of significance $\alpha = %s$, we fail to reject the null hypothesis' % alpha))
                print(
                    'At the %s level of significance, there is insufficient evidence to conclude that XXX and XXX are not independent.'%alpha)







            return results

    class Utils:
        def __init__(self):
            pass
        
        def test_against_5(number_to_test):
            if(number_to_test > 5):
                return(">")
            elif(number_to_test < 5) : 
                return("<")
            else:
                return("=")

    class Graphs:
        def __init__(self):
            pass

        class Normal:
            def __init__(self):
                pass

            def showGraph(z_values, type):

                # type == 'left'
                # type == 'right'
                # type == 'two'
                # type == 'between'


                plt.style.use('ggplot')
                mean = 0
                std = 1
                x = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
                iq = stats.norm(mean, std)
                plt.plot(x, iq.pdf(x), 'b')

                if type == 'left':
                    px_lt = np.arange(-3, z_values[0], .01)
                    plt.fill_between(px_lt, iq.pdf(px_lt), color='r')
                if type == 'right':
                    px_gt = np.arange(z_values[0], 3, .01)
                    plt.fill_between(px_gt, iq.pdf(px_gt), color='r')
                if type == 'two':
                    px_lt = np.arange(-3, z_values[0], .01)
                    plt.fill_between(px_lt, iq.pdf(px_lt), color='r')
                    px_gt = np.arange(z_values[1], 3, .01)
                    plt.fill_between(px_gt, iq.pdf(px_gt), color='r')
                if type == 'between':
                    px = np.arange(z_values[0], z_values[1], .01)
                    plt.fill_between(px, iq.pdf(px), color='r')






                plt.show()

        class Students:
            def __init__(self):
                pass

            def showGraph(t_values, dof, type):

                type == 'left'
                type == 'right'
                type == 'two'
                type == 'between'


                plt.style.use('ggplot')

                x = np.linspace(-10, 10, 1000)
                # iq = t.pdf(x, dof)
                plt.plot(x, t.pdf(x, dof), 'k')

                if type == 'left':
                    px_lt = np.arange(-10, t_values[0], .01)
                    plt.fill_between(px_lt, t.pdf(px_lt, dof), color='r')
                if type == 'right':
                    px_gt = np.arange(t_values[0], 10, .01)
                    plt.fill_between(px_gt, t.pdf(px_gt, dof), color='r')
                if type == 'two':
                    px_lt = np.arange(-10, t_values[0], .01)
                    plt.fill_between(px_lt, t.pdf(px_lt, dof), color='r')
                    px_gt = np.arange(t_values[1], 10, .01)
                    plt.fill_between(px_gt, t.pdf(px_gt, dof), color='r')
                if type == 'between':
                    px = np.arange(t_values[0], t_values[1], .01)
                    plt.fill_between(px, t.pdf(px, dof), color='r')






                plt.show()

        class Scatter:
            def __init__(self):
                pass

            def make_plot(x, y, trendline=True, meanpoint=False):

                plt.plot(x, y, 'o', color='red');
                z = np.polyfit(x, y, 1)

                if(trendline):
                    p = np.poly1d(z)
                    plt.plot(x, p(x), "black")


                if(meanpoint):

                    xpoint = MyStats.get_mean(x)
                    ypoint = MyStats.get_mean(y)

                    plt.plot(xpoint, ypoint, "bo")
                    # plt.text(xpoint, ypoint, r'$(\bar{x} , \bar{y})$', horizontalalignment='right', textcoords="offset points")

                    plt.annotate(r'$(\bar{x} , \bar{y})$',  # this is the text
                                 (xpoint, ypoint),  # this is the point to label
                                 textcoords="offset points",  # how to position the text
                                 xytext=(0, 10),  # distance from text to points (x,y)
                                 ha='center')  # horizontal alignment can be left, right or center

                plt.show()

                return plt


        class Histogram:
            def __init__(self):
                pass

            def make_plot(array, legend, x_labels, y_label,  title):

                a = array

                x = np.arange(len(x_labels))  # the label locations

                width = 0.5  # the width of the bars

                fig, ax = plt.subplots()
                # rects1 = ax.bar(x - width/2, a, width, label=a_label)
                rects1 = ax.bar(x, a, width, label=legend)


                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel(y_label)
                ax.set_title(title)
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels)
                ax.legend()

                def autolabel(rects):
                    """Attach a text label above each bar in *rects*, displaying its height."""
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate('{}'.format(height),
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 0),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom')

                autolabel(rects1)
                fig.tight_layout()
                fig.patch.set_facecolor('xkcd:white')

                plt.show()

    class HypothesisExamples:
        def __init__(self):
            pass

        class Reject:
            def __init__(self):
                pass

            class less:
                def __init__(self):
                    pass

                def get_examples(self):
                    return 'The sample evidence is sufficient at the %s level to justify rejecting the null hypothesis. It seems that ' \
                           '%s have a lower %s'

                    # return 'At the %s level of significance, sample data indicate that the average %s is less than %s'

                    #return 'At the %s level of significance, sample evidence supports the claim that %s is less than %s'

            class greater:
                def __init__(self):
                    pass

                def get_examples(self):
                    return 'The sample evidence is sufficient at the %s level to justify rejecting the null hypothesis. It seems that ' \
                           '%s have a greater %s'

                    # return 'At the %s level of significance, sample data indicate taht the average %s is greater than %s'
                    # return 'At the %s level of significance, sample evidence supports the claim that %s is greater than %s'

            class different:
                def __init__(self):
                    pass

                def get_examples(self):
                    return 'At the %s level of significance, we can say that the %s is different than %s'


        class FailToReject:
            def __init__(self):
                pass

            class less:
                def __init__(self):
                    pass

                def get_examples(self):
                    return 'At the %s level of significance, there is insufficient evidence to say %s is decreasing.'

            class greater:
                def __init__(self):
                    pass

                def get_examples(self):
                    return 'At the %s level of significance, there is insufficient evidence to say %s is increasing.'

            class different:
                def __init__(self):
                    pass

                def get_examples(self):
                    return 'There is insufficient evidence at the %s level of significance to reject the null hypothesis. It seems' \
                           'that average %s matches the %s average'

                    # return'At the %s level of significance , sample evidence does not support a claim that the average %s is different from thoat of those in %s'
                    # return 'At the %s level of significance, we cannot conclude that that the %s is different from %s'
                    #return 'At the %s level of significance, we cannot say that the %s is different than %s'