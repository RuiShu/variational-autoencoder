require 'KLDCriterion'
require 'GmmKLDCriterion'
require 'SimpleKLDCriterion'
require 'Sampler'
require 'IndGmmSampler'
require 'GmmSampler'
require 'grader'

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

function gmm_kld_check()
   local kld = nn.KLDCriterion()
   local gkld = nn.GmmKLDCriterion()
   local input = torch.randn(1,2,8)
   local target = torch.randn(1,2,8) + 1000
   local out = kld:forward(input, target)
   local gout = gkld:forward(input, target)
   print('kld '..out)
   print('gkld '..gout)
   -- dmulv, dpmulv = kld:backward(input, target)
   -- dgmulv, dgpmulv = gkld:backward(input, target)
   -- print((dmulv - dgmulv):abs():max())
   -- print((dpmulv - dgpmulv):abs():max())
end

-- require 'cunn'
function gmm_kld_grad_check()
   kld = nn.GmmKLDCriterion()

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
   input = torch.randn(10,2,1)
   target = torch.randn(10,100,1)
   grader.checkCriterionGradInput(kld, input, target)
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
   mu, lv = unpack(input:split(5,2))
   lv:fill(-1000)
   output = sampler:forward(input)
   print(mu)
   print(output)
end

function gmm_sampler_forward()
   sampler = nn.GmmSampler()
   input = torch.randn(5,6,3)
   mu, lv = unpack(input:split(3,2))
   for i = 1,3 do
      -- mu[{{},i}] = torch.Tensor({1,2,3}):view(1,3):expand(5,3)
      mu[{{},i}]:fill(i)
   end
   lv:fill(-1000)
   out = sampler:forward(input)
   print(out)
end

-- kld_grad_check()
-- ind_gmm_sampler_forward()
-- gmm_kld_check()
-- gmm_kld_grad_check()
gmm_sampler_forward()
