require 'KLDCriterion'
require 'SimpleKLDCriterion'
require 'grader'
-- require 'cunn'

kld = nn.KLDCriterion()
skld = nn.SimpleKLDCriterion()

-- test Gaussian
print("Gaussian")
input = torch.randn(2,5)
target = torch.randn(2,5)
grader.checkCriterionGradInput(kld, input, target)

-- test Gmm
print("Gmm")
input = torch.randn(2,5)
target = torch.randn(8,5)
grader.checkCriterionGradInput(kld, input, target)

-- test batch Gaussian
print("Batch Gaussian")
input = torch.randn(10,2,5)
target = torch.randn(10,2,5)
grader.checkCriterionGradInput(kld, input, target)

-- test batch Gmm
print("Batch Gmm")
input = torch.randn(10,2,5)
target = torch.randn(10,8,5)
grader.checkCriterionGradInput(kld, input, target)

-- speed test kld
print("Speed test KLD")
input = torch.randn(200,2,100)
target = torch.randn(200,2,100)
grader.speedTest(kld, input, target)

-- speed test skld
print("Speed test SKLD")
input = torch.randn(200,100)
target = torch.randn(200,100)
grader.speedTest(skld, input, target)
