require 'KLDCriterion'
require 'SimpleKLDCriterion'
require 'Sampler'
require 'IndGmmSampler'
require 'grader'
-- require 'cunn'

function kld_grad_check()
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
end

function kld_speed_test()
   kld = nn.KLDCriterion()
   skld = nn.SimpleKLDCriterion()

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
end

function sampler_forward()
   sampler = nn.Sampler()
   input = torch.randn(5,2,3)
   output = sampler:forward(input)
   print(output)
   input = torch.randn(1,2,3)
   sampler:forward(input)
   print(output)
end

function ind_gmm_sampler_forward()
   sampler = nn.IndGmmSampler()
   input = torch.randn(5,10,3)
   output = sampler:forward(input)
end

ind_gmm_sampler_forward()
