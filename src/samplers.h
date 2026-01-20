
// must be included after ncnn::Mat, SDCoroTask, SDCoroState, SDXLParams, sampler_type,
// CFGDenoiser_CompVisDenoiser and randn_4_w_h definitions

inline
void create_buffers(const sampler_type sampler,
                    std::vector<ncnn::Mat>& sampler_history_buffer,
                    const ncnn::Mat& x_mat)
{   // reserving memory for history elements
    unsigned number_of_buffers;
    switch (sampler) {
    case IPNDM: case IPNDM_V: case IPNDM_VO: case LMS:
        number_of_buffers = 4; break;
    case TAYLOR3: case DPMPP3MSDE: case DPMPP3MSDE_A:
        number_of_buffers = 3; break;
    case HEUN:
        number_of_buffers = 2; break;
    case DPMPP2S: case DPMPP2S_A: case DPMPP2M: case DPMPP2MV2: case DPM2:
        number_of_buffers = 1; break;
    default:
        number_of_buffers = 0;
    }
    for (unsigned k = 0; k < number_of_buffers; k++)
        sampler_history_buffer.push_back(ncnn::Mat(x_mat.w, x_mat.h, x_mat.c));
}

inline
void prescale_sample(ncnn::Mat& x_mat,
                     const sampler_type sampler,
                     const unsigned latent_length, // m_latw * m_lath
                     const int steps,
                     const int i, // step
                     std::vector<float>& sigma, // [steps + 1]
                     const bool m_turbo)
{
    // DDIM and TCD samplers implementation in SD.cpp needs latents prescaling
    // https://github.com/leejet/stable-diffusion.cpp/blob/10c6501bd05a697e014f1bee3a84e5664290c489/denoiser.hpp#L1071L1085
    switch (sampler) {
    case DDIM:
    case DDIM_A:
    case TCD:
    case TCD_A:
    if (i == 0) {
        for (int c = 0; c < 4; c++)
        {
            float* x_ptr = x_mat.channel(c); const float* x_ptr_end = x_ptr + latent_length;
            for (; x_ptr < x_ptr_end; x_ptr++)
                *x_ptr *= std::sqrt(sigma[i] * sigma[i] + 1) / sigma[i];
        }
    } else {
        float scale = std::sqrt(sigma[i] * sigma[i] + 1);
        if (m_turbo)                                       // soften correction for Turbo model
            scale = std::pow(scale, 0.9925f - 2.5f / steps / steps); // to avoid oversharpening
        for (int c = 0; c < 4; c++)
        {
            float* x_ptr = x_mat.channel(c); const float* x_ptr_end = x_ptr + latent_length;
            for (; x_ptr < x_ptr_end; x_ptr++)
                *x_ptr *= scale;
        }
    }
    default: ;
    }
}


#define ORIGINAL_SAMPLER_ALGORITHMS 1

#define ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS \
    float* x_ptr = x_mat.channel(c);                               \
    const float* x_ptr_end = x_ptr + latent_length;                \
    float* d_ptr = denoised.channel(c);

inline
SDCoroTask<int> process_sample( ncnn::Net& net,
                                int& seed,
                                const ncnn::Mat& c,
                                const ncnn::Mat& uc,
                                SDXLParams* sdxl_params,
                                SDCoroState& coro_state,
                                const float* log_sigmas,
                                ncnn::Mat& x_mat,
                                ncnn::Mat& denoised,
                                std::vector<ncnn::Mat>& sampler_history_buffer, // [0...4]
                                float& sampler_history_dt,                      // for Taylor3
                                float& eta, // for DDPM / DDIM / TCD / DPM++3M SDE: randomness
                                float (&lms_coeff)[4],                          // for LMS
                                const std::vector<float>& sigma,                // [steps + 1]
                                const unsigned latent_length,                   // m_latw * m_lath
                                const int steps,
                                const int i,                                    // current step
                                const bool m_xl,
                                const bool m_turbo,
                                const sampler_type sampler )
{
    // a workaround instead of configurable noise scheduler,
    // for Turbo model + non-ancestral samplers, use the result instead of sigma [i + 1]
    auto sigma_reshaper = [=](const float si1, const int i) -> const float {
        // only correct for turbo model
        if (!m_turbo) return si1;

        constexpr float p = 0.0f; // -n - smoother images, +n - sharper
        const float sigma_curve = ( std::pow((steps - i) / (float)steps, std::pow(2.f, -p - .5f) / steps) // straight, lower values at last steps
                                  + std::pow((i + 1)     / (float)steps, std::pow(2.f, -p - .5f) / steps) // reversed, lower values at first steps
                                  ) /2;
        return si1 * (sigma_curve ? std::max(0.0001f, sigma_curve) : 0.f);
    };

    // correcting sigmas less for many steps, or images become too blurry
    auto sigma_reshaper_sharp = [=](const float si1, const int i) -> const float {
        const float pre_res = sigma_reshaper(si1, i);
        const float smoothness = 3 / (steps - 2.5f);
        return si1 + ((pre_res == si1) ? 0 : smoothness / std::abs(smoothness) * std::pow(std::abs(smoothness), 1.f / 3) * (pre_res - si1));
    };

    switch (sampler) {
    case EULER: // Euler
    // adapted from https://github.com/leejet/stable-diffusion.cpp
    {
        const float si1 = sigma_reshaper(sigma[i + 1], i);
#if       ORIGINAL_SAMPLER_ALGORITHMS
        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            for (; x_ptr < x_ptr_end; x_ptr++)
                *x_ptr += (*x_ptr - *d_ptr++) / sigma[i] * (si1 - sigma[i]);
        }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
        const float sigma_mul = si1 / sigma[i];
        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                *x_ptr = (*x_ptr - *d_ptr) * sigma_mul + *d_ptr;
                d_ptr++;
            }
        }
#endif // ORIGINAL_SAMPLER_ALGORITHMS
        break;
    } // euler

    case HEUN: // Heun
    // adapted from https://github.com/leejet/stable-diffusion.cpp
    {
        const float si1 = sigma_reshaper(sigma[i + 1], i);
#if       ORIGINAL_SAMPLER_ALGORITHMS
        const float dt = si1 - sigma[i];
        if (!si1) { // Euler
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                for (; x_ptr < x_ptr_end; x_ptr++)
                    *x_ptr += (*x_ptr - *d_ptr++) / sigma[i] * dt;
            }
        } else {
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                float* b1_ptr = sampler_history_buffer[1].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    float d = (*x_ptr - *d_ptr++) / sigma[i];
                    *b0_ptr++ = *x_ptr + d * dt;
                    *b1_ptr++ = d;
                }
            }
            denoised = co_await CFGDenoiser_CompVisDenoiser(net, log_sigmas, sampler_history_buffer[0], si1, c, uc, sdxl_params, coro_state);
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                float* b1_ptr = sampler_history_buffer[1].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    float d2 = (*b0_ptr++ - *d_ptr++) / si1;
                    float d  = (*b1_ptr++ + d2) / 2;
                    *x_ptr += d * dt;
                }
            }
        }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
        if (!si1) { // pass
            x_mat = denoised;
        } else {
            const float sigma_div = 1.f / sigma[i];
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                float* b1_ptr = sampler_history_buffer[1].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    const float d = (*x_ptr - *d_ptr) * sigma_div;
                    *b0_ptr++ = *d_ptr++ + d * si1;
                    *b1_ptr++ = d;
                }
            }
            denoised = co_await CFGDenoiser_CompVisDenoiser(net, log_sigmas, sampler_history_buffer[0], si1, c, uc, sdxl_params, coro_state);
            const float sigma_d = (si1 - sigma[i]) / 2.f;
            const float sigma_inv = sigma_d / si1;
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                float* b1_ptr = sampler_history_buffer[1].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                    //*x_ptr += (*b0_ptr++ - *d_ptr++) * sigma_inv + *b1_ptr++ * sigma_d;
                    *x_ptr = *x_ptr + (*b0_ptr++ - *d_ptr++) * sigma_inv + *b1_ptr++ * sigma_d;
            }
        }
#endif // ORIGINAL_SAMPLER_ALGORITHMS
        break;
    } // heun

    case DPMPP2S: // DPM++ (2S)
    // adapted from https://github.com/leejet/stable-diffusion.cpp
    {
        const float si1 = sigma_reshaper(sigma[i + 1], i);
        if (!si1) { // no noise -> pass
            x_mat = denoised;
        } else {
            const float a = si1 / sigma[i];
            const float b = std::sqrt(a);
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {   // First half-step
                    *b0_ptr++ = *d_ptr + b * (*x_ptr - *d_ptr);
                    d_ptr++;
                }
            }
            denoised = co_await CFGDenoiser_CompVisDenoiser(net, log_sigmas, sampler_history_buffer[0], sigma[i + 1], c, uc, sdxl_params, coro_state);
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                for (; x_ptr < x_ptr_end; x_ptr++)
                {   // Second half-step
                    *x_ptr = *d_ptr + a * (*x_ptr - *d_ptr);
                    d_ptr++;
                }
            }
        }
        break;
    } // dpm++2s

    case DPMPP2S_A: // DPM++ (2S) Ancestral
    // adapted from https://github.com/leejet/stable-diffusion.cpp
    {
#if       ORIGINAL_SAMPLER_ALGORITHMS
        const float sigma_up = std::min(sigma[i + 1], std::sqrt(sigma[i + 1] * sigma[i + 1] * (sigma[i] * sigma[i] - sigma[i + 1] * sigma[i + 1]) / (sigma[i] * sigma[i])));
        const float sigma_down = std::sqrt(sigma[i + 1] * sigma[i + 1] - sigma_up * sigma_up);
        if (!sigma_down) { // pass
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                for (; x_ptr < x_ptr_end; x_ptr++)
                    *x_ptr = *d_ptr++;
            }
        } else {   // DPM-Solver++(2S)
            float t      = -std::log(sigma[i]);
            float t_next = -std::log(sigma_down);
            float h      = t_next - t;
            float s      = t + 0.5f * h;
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)   // First half-step
                    *b0_ptr++ = std::exp(-s) / std::exp(-t) * *x_ptr - std::expm1(-h * 0.5f) * *d_ptr++;
            }
            denoised = co_await CFGDenoiser_CompVisDenoiser(net, log_sigmas, sampler_history_buffer[0], sigma[i + 1], c, uc, sdxl_params, coro_state);
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                for (; x_ptr < x_ptr_end; x_ptr++)   // Second half-step
                    *x_ptr = (std::exp(-t_next) / std::exp(-t)) * *x_ptr - std::expm1(-h) * *d_ptr++;
            }
        }
        if (sigma[i + 1] > 0) {                      // Noise
            std::srand(seed++);
            ncnn::Mat randn = randn_4_w_h(rand() % 1000, g_main_args.m_latw, g_main_args.m_lath);

            for (int c = 0; c < 4; c++)
            {
                float* x_ptr = x_mat.channel(c); const float* x_ptr_end = x_ptr + latent_length;
                float* r_ptr = randn.channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                    *x_ptr += *r_ptr++ * sigma_up;
            }
        }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
        const float sigma_up = std::min(sigma[i + 1], 
            std::sqrt((sigma[i] + sigma[i + 1]) * (sigma[i] - sigma[i + 1])) * std::abs(sigma[i + 1] / sigma[i]));
        const float sigma_down = std::sqrt((sigma[i + 1] + sigma_up) * (sigma[i + 1] - sigma_up));
        if (!sigma_down) { // pass
            x_mat = denoised;
        } else {   // DPM-Solver++(2S)
            float a = sigma_down / sigma[i];
            float b = std::sqrt(a);
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {   // First half-step
                    *b0_ptr++ = *d_ptr + b * (*x_ptr - *d_ptr);
                    d_ptr++;
                }
            }
            denoised = co_await CFGDenoiser_CompVisDenoiser(net, log_sigmas, sampler_history_buffer[0], sigma[i + 1], c, uc, sdxl_params, coro_state);
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                for (; x_ptr < x_ptr_end; x_ptr++)
                {   // Second half-step
                    *x_ptr = *d_ptr + a * (*x_ptr - *d_ptr);
                    d_ptr++;
                }
            }
        }
        if (sigma[i + 1] > 0) {   // Noise
            std::srand(seed++);
            ncnn::Mat randn = randn_4_w_h(rand() % 1000, g_main_args.m_latw, g_main_args.m_lath);
            for (int c = 0; c < 4; c++)
            {
                float* x_ptr = x_mat.channel(c); const float* x_ptr_end = x_ptr + latent_length;
                float* r_ptr = randn.channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                    *x_ptr += *r_ptr++ * sigma_up;
            }
        }
#endif // ORIGINAL_SAMPLER_ALGORITHMS
        break;
    } // dpm++2s_a

    case DPMPP2M: // DPM++ (2M)
    // adapted from https://github.com/leejet/stable-diffusion.cpp
    {
        const float si1 = sigma_reshaper(sigma[i + 1], i);
#if       ORIGINAL_SAMPLER_ALGORITHMS
        if (!i || !si1) {
            float a = si1 / sigma[i];
            float b = std::expm1(std::log(si1) - std::log(sigma[i]));
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    *x_ptr = a * *x_ptr - b * *d_ptr;
                    *b0_ptr++ = *d_ptr++;
                }
            }
        } else {
            float t       = -std::log(sigma[i]);
            float t_next  = -std::log(si1);
            float h       = t_next - t;
            float a       = si1 / sigma[i];
            float b       = std::expm1(-h);
            float h_last  = t + std::log(sigma[i - 1]);
            float r       = h_last / h;
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    float d = (1.f + 1.f / (2.f * r)) * *d_ptr 
                                  - (1.f / (2.f * r)) * *b0_ptr;
                    *x_ptr = a * *x_ptr - b * d;
                    *b0_ptr++ = *d_ptr++;
                }
            }
        }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
        const float sigma_mul = si1 / sigma[i];
        if (!i || !si1) {   // Euler step
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    *x_ptr = (*x_ptr - *d_ptr) * sigma_mul + *d_ptr;
                    *b0_ptr++ = *d_ptr++;
                }
            }
        } else {
            float b = sigma_mul - 1.f;
            float r = .5f * std::log(sigma_mul) / std::log(sigma[i] / sigma[i - 1]) * b;
            b = -(b + r);

            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    *x_ptr = sigma_mul * *x_ptr + b * *d_ptr + r * *b0_ptr;
                    *b0_ptr++ = *d_ptr++;
                }
            }
        }
#endif // ORIGINAL_SAMPLER_ALGORITHMS
        break;
    } // dpm++2m

    // DPM++ 3M SDE, needs more than 100 steps, otherwise it produces significant noise
    case DPMPP3MSDE_A:
// TODO: maybe use sigma_down + sigma_up instead of eta?
        if (!g_main_args.m_turbo)  // SD / SDXL accept high noise
            eta = 1.0f;
        else            // Turbo model does not seem to work well with SDE + high noise,
            eta = 0.5f; // images are smudgy
    case DPMPP3MSDE:
    // adapted from https://github.com/crowsonkb/k-diffusion
    {
        // shifting 2 history elements
        if (i) for (int k = 2; k; k--) sampler_history_buffer[k] = sampler_history_buffer[k - 1];

#if       ORIGINAL_SAMPLER_ALGORITHMS
        // corrections for Turbo model, "no op" for SD / SDXL
        float si1 =              sigma_reshaper(sigma[i + 1], i),
            si0 = (i == 0) ? 1 : sigma_reshaper(sigma[i],     i - 1),
            sm1 = (i <= 1) ? 1 : sigma_reshaper(sigma[i - 1], i - 2);
        // it seems that single correction is not enough
        si1 = (si1 +                 sigma_reshaper(si1, i)     ) / 2;
        si0 = (si0 + ((i == 0) ? 1 : sigma_reshaper(si0, i - 1))) / 2;
        sm1 = (sm1 + ((i <= 1) ? 1 : sigma_reshaper(sm1, i - 2))) / 2;

        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            float *b0_ptr = sampler_history_buffer[0].channel(c),
                  *b1_ptr = sampler_history_buffer[1].channel(c),
                  *b2_ptr = sampler_history_buffer[2].channel(c);
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                const float d = *d_ptr; *b0_ptr = d; // put to history
                if (!si1) {
                    *x_ptr = d; // last step = pure denoising
                } else if (i > 1) {
                    const float h = std::log(sigma[i])     - std::log(si1),
                        h_1       = std::log(sigma[i - 1]) - std::log(si0),
                        h_2       = std::log(sigma[i - 2]) - std::log(sm1),
                        h_eta = h * (eta + 1);
                    *x_ptr = std::exp(-h_eta) * *x_ptr - std::expm1(-h_eta) * d;
                    const float r = h_1 / h,
                        r2 = h_2 / h,
                        d1_0 = (d - *b1_ptr) / r,
                        d1_1 = (*b1_ptr - *b2_ptr) / r2,
                        d1 = d1_0 + (d1_0 - d1_1) * r / (r + r2),
                        d2 = (d1_0 - d1_1) / (r + r2),
                        phi_2 = std::expm1(-h_eta) / h_eta + 1,
                        phi_3 = phi_2 / h_eta - 0.5f;
                    *x_ptr += phi_2 * d1 - phi_3 * d2;
                } else if (i) {
                    const float h = std::log(sigma[i])     - std::log(si1),
                        h_1       = std::log(sigma[i - 1]) - std::log(si0),
                        h_eta = h * (eta + 1);
                    *x_ptr = std::exp(-h_eta) * *x_ptr - std::expm1(-h_eta) * d;
                    const float r = h_1 / h,
                        phi_2 = std::expm1(-h_eta) / h_eta + 1,
                        d1 = (d - *b1_ptr) / r;
                    *x_ptr += phi_2 * d1;
                } else {
                    const float h = std::log(sigma[i]) - std::log(si1),
                        h_eta = h * (eta + 1);
                    *x_ptr = std::exp(-h_eta) * *x_ptr - std::expm1(-h_eta) * d;
                }
                d_ptr++; b0_ptr++; b1_ptr++; b2_ptr++;
            }
        }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
        //long double
        //double
        float
        si1 = sigma_reshaper(sigma[i + 1], i), si0, sm1, k1, k2, k3, k4;
        si1 = (si1 + sigma_reshaper(si1, i)) / 2;

        if (si1) {
            k1 = std::pow(si1 / sigma[i], eta + 1); // exp(-h_eta) in k-diffusion
            k2 = 1 - k1;
            if (i) {
                si0 = sigma_reshaper(sigma[i], i - 1);
                si0 = (si0 + ((i == 0) ? 1 : sigma_reshaper(si0, i - 1))) / 2;
                const auto h  = si1 ? std::log(sigma[i]     / si1) : 1,
                           h1 = si0 ? std::log(sigma[i - 1] / si0) : 1,
                           r = 1 / (eta + h / h); // 1 / (eta + 1)
                if (i == 1) {
                    k2 = (k2 * r * eta + h) / h1;
                    k3 = 1 - k1 - k2;
                } else {
                    sm1 = sigma_reshaper(sigma[i - 1], i - 2);
                    sm1 = (sm1 + ((i <= 1) ? 1 : sigma_reshaper(sm1, i - 2))) / 2;
                    const auto h2 = sm1 ? std::log(sigma[i - 2] / sm1) : 1,
                               h3 = h1 + h2;
                    k3 = (k2 * r * (h3 - r) + h * (r - h3 - 0.5f * h)) / h1 / h2; // using k2 = 1 - k1
                    k2 = (h - k2 * r) / h3 * (2 + h2 / h1 - r / h1) + k2 + h / h3 * (0.5f * h / h1);
                    k4 = 1 - k1 - k2 - k3;
                }
            }
        }

        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            float *b0_ptr = sampler_history_buffer[0].channel(c),
                  *b1_ptr = sampler_history_buffer[1].channel(c),
                  *b2_ptr = sampler_history_buffer[2].channel(c);
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                const float d = *d_ptr++; *b0_ptr++ = d; // put to history
                if (!si1) {
                    *x_ptr = d; // last step = pure denoising
                } else switch (i) {
                case 0:  *x_ptr = k1 * *x_ptr + k2 * d; break;
                case 1:  *x_ptr = k1 * *x_ptr + k2 * d + k3 * *b1_ptr++; break;
                default: *x_ptr = k1 * *x_ptr + k2 * d + k3 * *b1_ptr++ + k4 * *b2_ptr++;
                }
            }
        }
#endif // ORIGINAL_SAMPLER_ALGORITHMS
        if (eta && si1) {
            std::srand(seed++);
            ncnn::Mat randn = randn_4_w_h(rand() % 1000, g_main_args.m_latw, g_main_args.m_lath);
            const auto autozero = si1 * 0,
                variance = si1 * std::sqrt(std::max(autozero, 1 - std::pow(si1 / sigma[i], 2 * eta)));
            for (int c = 0; c < 4; c++)
            {
                float* x_ptr = x_mat.channel(c); const float* x_ptr_end = x_ptr + latent_length;
                float* r_ptr = randn.channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                    *x_ptr += *r_ptr++ * variance;
            }
        }
        break;
    } // dpm++3msde

    case DPMPP2MV2: // DPM++ (2M) v2
    // adapted from https://github.com/leejet/stable-diffusion.cpp
    {
        const float si1 = sigma_reshaper_sharp(sigma[i + 1], i);
#if       ORIGINAL_SAMPLER_ALGORITHMS
        if (!i || !si1) {
            float a = si1 / sigma[i];
            float b = std::expm1(std::log(si1) - std::log(sigma[i]));
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    *x_ptr = a * *x_ptr - b * *d_ptr;
                    *b0_ptr++ = *d_ptr++;
                }
            }
        } else {
            float t       = -std::log(sigma[i]);
            float t_next  = -std::log(si1);
            float h       = t_next - t;
            float a       = si1 / sigma[i];
            float h_last  = t + std::log(sigma[i - 1]);
            float h_min   = std::min(h_last, h);
            float h_max   = std::max(h_last, h);
            float r       = h_max / h_min;
            float h_d    = (h_max + h_min) / 2.f;
            float b      = std::expm1(-h_d);
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    float d = (1.f + 1.f / (2.f * r)) * *d_ptr 
                                  - (1.f / (2.f * r)) * *b0_ptr;
                    *x_ptr = a * *x_ptr - b * d;
                    *b0_ptr++ = *d_ptr++;
                }
            }
        }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
        // simplified
        if (!i || !si1) { // Euler step
            const float sigma_mul = si1 / sigma[i];
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    *x_ptr = (*x_ptr - *d_ptr) * sigma_mul + *d_ptr;
                    *b0_ptr++ = *d_ptr++;
                }
            }
        } else {
            float a       = si1 / sigma[i];
            float a_last  = sigma[i] / sigma[i - 1];
            float a_max   = std::max(a_last, a);
            float a_min   = std::min(a_last, a);
            float r       = std::log(a_max) / std::log(a_min) / 2.f;
            float m       = 1.f - std::sqrt(a_min * a_max);
            r *= -m;
            m -= r;
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    *x_ptr = a * *x_ptr + *d_ptr * m + *b0_ptr * r;
                    *b0_ptr++ = *d_ptr++;
                }
            }
        }
#endif // ORIGINAL_SAMPLER_ALGORITHMS
        break;
    } // dpm++2mv2

    case DPM2: // DPM2
    // adapted from https://github.com/leejet/stable-diffusion.cpp
    {
        const float si1 = sigma_reshaper(sigma[i + 1], i);
#if       ORIGINAL_SAMPLER_ALGORITHMS
        if (!si1) { // pass
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                for (; x_ptr < x_ptr_end; x_ptr++)
                    *x_ptr = *d_ptr++;
            }
        } else { // DPM-Solver-2
            float sigma_mid = std::exp(0.5f * (std::log(sigma[i]) + std::log(si1)));
            float dt_1      = sigma_mid - sigma[i];
            float dt_2      = si1 - sigma[i];
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    float d = (*x_ptr - *d_ptr++) / sigma[i];
                    *b0_ptr++ = *x_ptr + d * dt_1;
                }
            }
            denoised = co_await CFGDenoiser_CompVisDenoiser(net, log_sigmas, sampler_history_buffer[0], sigma_mid, c, uc, sdxl_params, coro_state);
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    float d2 = (*b0_ptr++ - *d_ptr++) / sigma_mid;
                    *x_ptr += d2 * dt_2;
                }
            }
        }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
        if (!si1) { // pass
            x_mat = denoised;
        } else { // DPM-Solver-2
            float sigma_mid = std::sqrt(sigma[i] * si1);
            float dt = sigma_mid / sigma[i] - 1.f;
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                    *b0_ptr++ = *x_ptr + (*x_ptr - *d_ptr++) * dt;
            }
            denoised = co_await CFGDenoiser_CompVisDenoiser(net, log_sigmas, sampler_history_buffer[0], sigma_mid, c, uc, sdxl_params, coro_state);
            dt = (si1 - sigma[i]) / sigma_mid;
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* b0_ptr = sampler_history_buffer[0].channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                    *x_ptr += (*b0_ptr++ - *d_ptr++) * dt;
            }
        }
#endif // ORIGINAL_SAMPLER_ALGORITHMS
        break;
    } // dpm2

    case IPNDM:  // iPNDM sampler from https://github.com/zju-pi/diff-sampler/tree/main/diff-solvers-main
    // adapted from https://github.com/leejet/stable-diffusion.cpp
    {
        const float si1 = sigma_reshaper(sigma[i + 1], i);
        float sd = si1 - sigma[i];

        // shifting 3 history elements
        if (i) for (int k = 3; k; k--) sampler_history_buffer[k] = sampler_history_buffer[k - 1];

        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            float *b0_ptr = sampler_history_buffer[0].channel(c),
                  *b1_ptr = sampler_history_buffer[1].channel(c),
                  *b2_ptr = sampler_history_buffer[2].channel(c),
                  *b3_ptr = sampler_history_buffer[3].channel(c);
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                float d = (*x_ptr - *d_ptr++) / sigma[i]; *b0_ptr++ = d;
                switch (i) {
                case 0: // first, Euler step
                    *x_ptr += sd * d;
                    break;
                case 1: // second, use one history point
                    *x_ptr += sd * (3 * d  - *b1_ptr++) / 2;
                    break;
                case 2: // third, use two history points
                    *x_ptr += sd * (23 * d - 16 * *b1_ptr++ + 5 * *b2_ptr++) / 12;
                    break;
                default: // fourth+, use three history points
                    *x_ptr += sd * (55 * d - 59 * *b1_ptr++ + 37 * *b2_ptr++ - 9 * *b3_ptr++) / 24;
                }
            }
        }
        break;
    } // ipndm

    case IPNDM_V:  // iPNDM_v sampler from https://github.com/zju-pi/diff-sampler/tree/main/diff-solvers-main
    // adapted from https://github.com/leejet/stable-diffusion.cpp
    {
        const float si1 = sigma_reshaper(sigma[i + 1], i);
        float h_n = si1 - sigma[i];
        float h_n_1 = (i > 0) ? (sigma[i] - sigma[i - 1]) : h_n;

        // shifting 3 history elements
        if (i) for (int k = 3; k; k--) sampler_history_buffer[k] = sampler_history_buffer[k - 1];

        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            float *b0_ptr = sampler_history_buffer[0].channel(c),
                  *b1_ptr = sampler_history_buffer[1].channel(c),
                  *b2_ptr = sampler_history_buffer[2].channel(c),
                  *b3_ptr = sampler_history_buffer[3].channel(c);
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                float d = (*x_ptr - *d_ptr++) / sigma[i]; *b0_ptr++ = d;
                switch (i) {
                case 0: // first, Euler step
                    *x_ptr += h_n * d;
                    break;
                case 1: // second, use one history point
                    *x_ptr += h_n * ((2 + (h_n / h_n_1)) * d - (h_n / h_n_1) * *b1_ptr++) / 2;
                    break;
                case 2: // third, use two history points
                    *x_ptr += h_n * (23 * d - 16 * *b1_ptr++ + 5 * *b2_ptr++) / 12;
                    break;
                default: // fourth+, use three history points
                    *x_ptr += h_n * (55 * d - 59 * *b1_ptr++ + 37 * *b2_ptr++ - 9 * *b3_ptr++) / 24;
                }
            }
        }
        break;
    } // ipndm_v

    case IPNDM_VO:  // iPNDM_v sampler from https://github.com/zju-pi/diff-sampler/tree/main/diff-solvers-main
    // replicated Python code of variable-step version, without modifications
    // slower, needs ~2 times more steps
    {
        const float si1 = sigma_reshaper(sigma[i + 1], i);

        // shifting 3 history elements
        if (i) for (int k = 3; k; k--) sampler_history_buffer[k] = sampler_history_buffer[k - 1];
        // not needed, array is copied
        //sampler_history_buffer[0] = ncnn::Mat(x_mat.w, x_mat.h, x_mat.c);

#if       ORIGINAL_SAMPLER_ALGORITHMS
        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            float *b0_ptr = sampler_history_buffer[0].channel(c),
                  *b1_ptr = sampler_history_buffer[1].channel(c),
                  *b2_ptr = sampler_history_buffer[2].channel(c),
                  *b3_ptr = sampler_history_buffer[3].channel(c);
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                float d = (*x_ptr - *d_ptr++) / sigma[i]; *b0_ptr++ = d;
                switch (i) {
                case 0: // first, Euler step
                {
                    float h_n = si1 - sigma[i];
                    *x_ptr += h_n * d;
                    break;
                }
                case 1: // second, use one history point
                {
                    float h_n = (si1 - sigma[i]);
                    float h_n_1 = (sigma[i] - sigma[i-1]);
                    float coeff1 = (2 + (h_n / h_n_1)) / 2;
                    float coeff2 = -(h_n / h_n_1) / 2;
                    *x_ptr += h_n * (coeff1 * d + coeff2 * *b1_ptr++);
                    break;
                }
                case 2: // third, use two history points
                {
                    float h_n = (si1 - sigma[i]);
                    float h_n_1 = (sigma[i] - sigma[i-1]);
                    float h_n_2 = (sigma[i-1] - sigma[i-2]);
                    float temp = (1 - h_n / (3 * (h_n + h_n_1)) 
                               * (h_n * (h_n + h_n_1)) 
                               / (h_n_1 * (h_n_1 + h_n_2))) / 2;
                    float coeff1 = (2 + (h_n / h_n_1)) / 2 + temp;
                    float coeff2 = -(h_n / h_n_1) / 2 
                                 - (1 + h_n_1 / h_n_2) * temp;
                    float coeff3 = temp * h_n_1 / h_n_2;
                    *x_ptr += (si1 - sigma[i]) 
                           * (coeff1 * d + coeff2 
                           * *b1_ptr++ 
                           + coeff3 * *b2_ptr++);
                    break;
                }
                default: // fourth+, use three history points
                {
                    float h_n = (si1 - sigma[i]);
                    float h_n_1 = (sigma[i] - sigma[i-1]);
                    float h_n_2 = (sigma[i-1] - sigma[i-2]);
                    float h_n_3 = (sigma[i-2] - sigma[i-3]);
                    float temp1 = (1 - h_n / (3 * (h_n + h_n_1)) 
                                * (h_n * (h_n + h_n_1)) 
                                / (h_n_1 * (h_n_1 + h_n_2))) / 2;
                    float temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) 
                                / 2 + (1 - h_n / (2 * (h_n + h_n_1))) 
                                * h_n / (6 * (h_n + h_n_1 + h_n_2))) 
                                * (h_n * (h_n + h_n_1) * (h_n + h_n_1 
                                + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) 
                                * (h_n_1 + h_n_2 + h_n_3));
                    float coeff1 = (2 + (h_n / h_n_1)) / 2 + temp1 
                                 + temp2;
                    float coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 
                                 / h_n_2) * temp1 - (1 + (h_n_1 
                                 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2)
                                 / (h_n_2 * (h_n_2 + h_n_3)))) * temp2;
                    float coeff3 = temp1 * h_n_1 / h_n_2 + ((h_n_1 
                                 / h_n_2) 
                                 + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 
                                 * (h_n_2 + h_n_3))) 
                                 * (1 + h_n_2 / h_n_3)) * temp2;
                    float coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) 
                                 / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 
                                 / h_n_2;
                    *x_ptr += (si1 - sigma[i]) * (coeff1 * d 
                            + coeff2 * *b1_ptr++ 
                            + coeff3 * *b2_ptr++
                            + coeff4 * *b3_ptr++);
                } // 4+
                } // switch()
            }
        }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
        float
        //double
        //long double
                   h_n = si1 - sigma[i], 
                   h_n_1, h_n_2, h_n_3, 
                   coeff11, coeff12,
                   coeff21, coeff22, coeff23,
                   coeff31, coeff32, coeff33, coeff34;
        if (i)     {
                   h_n_1 = sigma[i] - sigma[i - 1];
                   coeff11 = -h_n / h_n_1 * h_n / 2;
                   coeff12 = h_n - coeff11;
        }
        if (i > 1) {
                   h_n_2 = sigma[i - 1] - sigma[i - 2];
                   auto temp = .5f - h_n / h_n_1 * h_n / (h_n_1 + h_n_2) / 6;
                   coeff21 = h_n * ((2 + (h_n / h_n_1)) / 2 + temp);
                   coeff22 = h_n * (-(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp);
                   coeff23 = h_n * temp * h_n_1 / h_n_2;
        }
        if (i > 2) {
                   h_n_3 = sigma[i - 2] - sigma[i - 3];
                   auto temp2 = (.5f - h_n / (h_n + h_n_1) / 6
                                + (h_n_1 + si1 - sigma[i - 1]) / (h_n + h_n_1)
                                 * h_n / (h_n + h_n_1 + h_n_2) / 12
                                )
                                *  h_n                  /  h_n_1
                                * (h_n + h_n_1        ) / (h_n_1 + h_n_2        )
                                * (h_n + h_n_1 + h_n_2) / (h_n_1 + h_n_2 + h_n_3);
                   coeff31 = coeff21 + h_n * temp2;
                   coeff32 = coeff22 - h_n * ((1 + (h_n_1 
                           / h_n_2) + (h_n_1 * (h_n_1 + h_n_2)
                           / (h_n_2 * (h_n_2 + h_n_3)))) * temp2);
                   coeff33 = coeff23 + h_n * (((h_n_1 
                           / h_n_2) 
                           + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 
                           * (h_n_2 + h_n_3))) 
                           * (1 + h_n_2 / h_n_3)) * temp2);
                   coeff34 = h_n * (-temp2 * (h_n_1 * (h_n_1 + h_n_2) 
                           / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 
                           / h_n_2);
        }

        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            float *b0_ptr = sampler_history_buffer[0].channel(c),
                  *b1_ptr = sampler_history_buffer[1].channel(c),
                  *b2_ptr = sampler_history_buffer[2].channel(c),
                  *b3_ptr = sampler_history_buffer[3].channel(c);
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                float d = (*x_ptr - *d_ptr++) / sigma[i]; *b0_ptr++ = d;
                switch (i) {
                case 0: // first, Euler step
                {
                    *x_ptr += h_n * d;
                    break;
                }
                case 1: // second, use one history point
                {
                    *x_ptr +=  coeff12 * d + coeff11 * *b1_ptr++;
                    break;
                }
                case 2: // third, use two history points
                {
                    *x_ptr += coeff21 * d 
                            + coeff22 * *b1_ptr++ 
                            + coeff23 * *b2_ptr++;
                    break;
                }
                default: // fourth+, use three history points
                {
                    *x_ptr += coeff31 * d 
                            + coeff32 * *b1_ptr++ 
                            + coeff33 * *b2_ptr++
                            + coeff34 * *b3_ptr++;
                } // 4+
                } // switch()
            }
        }
#endif // ORIGINAL_SAMPLER_ALGORITHMS
        break;
    } // ipndm_vo

    case TAYLOR3:  // third-order-Taylor extension of Euler
    // adapted from https://github.com/aagdev/mlimgsynth
    {
        const float si1 = sigma_reshaper_sharp(sigma[i + 1], i);
#if       ORIGINAL_SAMPLER_ALGORITHMS
        float dt = si1 - sigma[i], idtp, f2, f3, d2, d3;
        idtp = 1 / sampler_history_dt;
        f2 = dt * dt / 2;
        f3 = dt * dt * dt / 6;

        // shifting 2 history elements
        if (i) for (int k = 2; k; k--) sampler_history_buffer[k] = sampler_history_buffer[k - 1];

        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            float *b0_ptr = sampler_history_buffer[0].channel(c),
                  *b1_ptr = sampler_history_buffer[1].channel(c),
                  *b2_ptr = sampler_history_buffer[2].channel(c);
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                float d = (*x_ptr - *d_ptr++) / sigma[i]; *b0_ptr++ = d;
                switch (i) {
                case 0: // first, Euler step
                {
                    *x_ptr += dt * d;
                    break;
                }
                case 1: // second, using one history point
                {
                    d2 = (d - *b1_ptr++) * idtp;
                    *x_ptr += dt * d + f2 * d2;
                    break;
                }
                default: // third+, using two history points
                {
                    d2 = (d - *b1_ptr++) * idtp;
                    d3 = (d2 - *b2_ptr++) * idtp;
                    *x_ptr += dt * d + f2 * d2 + f3 * d3;
                    break;
                } // 3+
                } // switch()
            }
        }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
        float
        //double
        //long double
            dt = si1 - sigma[i], f11, f12, f21, f22, f23;
        if (i) {
            f12 = -dt * dt / sampler_history_dt / 2;
            f11 = dt - f12;
            if (i > 1) {
                f23 = f12 * dt / 3;
                f22 = f12 + f23 / sampler_history_dt;
                f21 = dt - f22;
            }
        }

        // shifting 2 history elements
        if (i) for (int k = 2; k; k--) sampler_history_buffer[k] = sampler_history_buffer[k - 1];

        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            float *b0_ptr = sampler_history_buffer[0].channel(c),
                  *b1_ptr = sampler_history_buffer[1].channel(c),
                  *b2_ptr = sampler_history_buffer[2].channel(c);
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                float d = (*x_ptr - *d_ptr++) / sigma[i]; *b0_ptr++ = d;
                switch (i) {
                case 0: // first, Euler step
                {
                    *x_ptr += dt * d;
                    break;
                }
                case 1: // second, using one history point
                {
                    *x_ptr += f11 * d + f12 * *b1_ptr++;
                    break;
                }
                default: // third+, using two history points
                {
                    *x_ptr += f21 * d + f22 * *b1_ptr++ + f23 * *b2_ptr++;
                    break;
                } // 3+
                } // switch()
            }
        }
#endif // ORIGINAL_SAMPLER_ALGORITHMS

        sampler_history_dt = dt;

        break;
    } // taylor3

    case DDPM_A: // Denoising Diffusion Probabilistic Models
        eta = 1.f; // random == ancestral
    case DDPM:
    // adapted from https://github.com/Windsander/ADI-Stable-Diffusion
    {
        float s2 = sigma[i] * sigma[i];
        float sn2 = sigma[i + 1] * sigma[i + 1];
        float scale_back = std::sqrt(s2 + 1.0f);
        float d = std::sqrt(sn2 + 1.0f);
        float variance = (eta <= 0) ? 0.0f :
                         (eta * std::sqrt(s2 - sn2) / d * sigma[i + 1] / sigma[i]);
        float a = sn2 / s2 * scale_back / d;
        float b = (s2 - sn2) / d / s2;

        if (variance <= 0) {
        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                *x_ptr = *x_ptr * a + *d_ptr++ * b;
            }
        }
        } else { // variance > 0
        std::srand(seed++);
        ncnn::Mat randn = randn_4_w_h(rand() % 1000, g_main_args.m_latw, g_main_args.m_lath);
        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            float* r_ptr = randn.channel(c);
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                *x_ptr = *x_ptr * a + *d_ptr++ * b + *r_ptr++ * variance;
            }
        }
        }
        break;
    } // ddpm / ddpm_a

    case DDIM: // Denoising Diffusion Implicit Models
    // adapted from https://github.com/leejet/stable-diffusion.cpp,
    // simplified (without eta)
    {
        const float si1 = sigma_reshaper_sharp(sigma[i + 1], i);

        //float
        double
        //long double
        const sn2 = si1 * si1,
              alpha_prod_t_prev = 1 / (sn2 + 1),
              a = std::sqrt(1 - alpha_prod_t_prev) / sigma[i],
              b = std::sqrt(alpha_prod_t_prev) - a;
        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                *x_ptr = *x_ptr * a + *d_ptr++ * b;
            }
        }
        break;
    } // ddim

    case DDIM_A: // Denoising Diffusion Implicit Models
    // adapted from https://github.com/leejet/stable-diffusion.cpp,
    {
        eta = 1.f; // randomness
        //long double
        //double
        float
        const si1 = sigma_reshaper_sharp(sigma[i + 1], i),
            alpha_prod_t      = 1 / (sigma[i] * sigma[i] + 1),
            alpha_prod_t_prev = 1 / (si1 * si1 + 1),
            beta_prod_t       = 1 - alpha_prod_t,
            beta_prod_t_prev  = 1 - alpha_prod_t_prev,
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev),
            autozero = variance * 0,
            std_dev_t = eta * std::sqrt(std::max(autozero, variance));
#if       ORIGINAL_SAMPLER_ALGORITHMS
        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                const auto model_output = (autozero + *x_ptr - *d_ptr++) / sigma[i],
                pred_original_sample = (*x_ptr * std::sqrt(alpha_prod_t)
                    - model_output * std::sqrt(beta_prod_t)) / std::sqrt(alpha_prod_t),
                pred_sample_direction = model_output
                    * std::sqrt(1 - alpha_prod_t_prev - variance * eta * eta);
                *x_ptr = std::sqrt(alpha_prod_t_prev) * pred_original_sample 
                    + pred_sample_direction;
            }
        }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
        const auto k0 = si1 / sigma[i],
            k1 = std::sqrt(std::max(autozero, 1 + (eta + k0 * eta) * (k0 * eta - eta))) * k0,
            a = k1 * std::sqrt(alpha_prod_t_prev),
            b = std::sqrt(alpha_prod_t_prev) - a;
        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                *x_ptr = *x_ptr * a + *d_ptr++ * b;
            }
        }
#endif // ORIGINAL_SAMPLER_ALGORITHMS
        if (eta > 0) {
            std::srand(seed++);
            ncnn::Mat randn = randn_4_w_h(rand() % 1000, g_main_args.m_latw, g_main_args.m_lath);
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* r_ptr = randn.channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                    *x_ptr += *r_ptr++ * std_dev_t;
            }
        }
        break;
    } // ddim_a

    case TCD_A: // Trajectory Consistency Distillation
        eta = 0.5f; // randomness, 1.0 also works
    case TCD:
    // adapted from https://github.com/leejet/stable-diffusion.cpp,
    // simplified, alphas are derived from sigmas and therefore are not smooth
    {
        //long double
        //double
        float
        const si = sigma[i],
              si1 = sigma_reshaper_sharp(sigma[i + 1], i),
              // sigma with scaled value == following1, smooth values (relative to eta),
              si4 = si1 * (1 - eta), // might not work with non-exponential / linear schedulers
              // sigma with scaled index == following2, discrete values (relative to eta)
              si3 = sigma[((steps - i - 1) * eta) + i + 1],
              // mixing sigmas to smoothen alpha_s
              si2 = std::sqrt(std::sqrt(si3 * (sigma[i + 1] ? si3 * (si1 / sigma[i + 1]) : si3)) *
                         std::sqrt(si4 * std::sqrt(si3 * si4))),
// TODO: maybe it's possible to interpolate between sigma[i*eta] and sigma[(i+1)*eta] ?
              alpha_n = 1 / (si1 * si1 + 1),
              alpha_s = 1 / (si2 * si2 + 1),
#if       ORIGINAL_SAMPLER_ALGORITHMS
              alpha   = 1 / (si  * si  + 1),
              beta    = 1 - alpha,
              beta_s  = 1 - alpha_s;
        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                const auto model_output = (*x_ptr - *d_ptr++) / si;
                const auto pred_original_sample = *x_ptr - std::sqrt(beta) / std::sqrt(alpha) * model_output;
                *x_ptr = std::sqrt(alpha_s) * pred_original_sample + std::sqrt(beta_s) * model_output;
            }
        }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
              am = si2 / si * std::sqrt(alpha_s),
              bm = std::sqrt(alpha_s) - am; // (1 - si2 / si) * sqrt(alpha_s)
        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                *x_ptr = *x_ptr * am + *d_ptr++ * bm;
            }
        }
#endif // ORIGINAL_SAMPLER_ALGORITHMS
        if (eta > 0 && i < (steps - 1)) {
            std::srand(seed++);
            ncnn::Mat randn = randn_4_w_h(rand() % 1000, g_main_args.m_latw, g_main_args.m_lath);
            const auto a = std::sqrt(alpha_n / alpha_s),
            // alpha_s can be smaller than alpha_n because of sigma_reshaper() or different noise scheduler,
            // therefore trimming negative differences
            autozero = si * 0, b = std::sqrt(std::max(autozero, 1 - alpha_n / alpha_s));
            for (int c = 0; c < 4; c++)
            {
                float* x_ptr = x_mat.channel(c); const float* x_ptr_end = x_ptr + latent_length;
                float* r_ptr = randn.channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                    *x_ptr = a * *x_ptr + b * *r_ptr++;
            }
        }
        break;
    } // tcd / tcd_a

    case LMS: // Linear Multi-Step
    // adapted from https://github.com/crowsonkb/k-diffusion
    // integrator is adapted from https://github.com/Windsander/ADI-Stable-Diffusion
    {
#if       ORIGINAL_SAMPLER_ALGORITHMS
        auto linear_multistep_coeff = [=](const int order, const int m, const int j) -> float {
            auto fn = [=](const float tau) -> const float {
                float prod = 1.f;
                for (int k = 0; k < order; k++)
                    if (j != k) prod *= (tau - sigma[m - k]) / (sigma[m - j] - sigma[m - k]);
                return prod;
            };
            constexpr int n = 30720; // 1 x 2 x 3 x 4 x 5 x 256
            auto simpson_integral = [=](const float a, const float b) -> float {
                // Simpson integral
                const float h = (b - a) / n;
                float sum = fn(a) + fn(b);
                for (int i = 1; i < n; i += 2) {
                    sum += 4 * fn(a + i * h);
                }
                for (int i = 2; i < n - 1; i += 2) {
                    sum += 2 * fn(a + i * h);
                }
                return sum * h / 3.0f;
            };
            auto simpson38_integral = [=](const float a, const float b) -> float {
                // Simpson 3/8 integral
                const float h = (b - a) / n;
                float sum = 0.f;
                for (int i = 0; i < n - 1; i += 3) {
                    const float s = a + h * i;
                    sum +=  fn(s) 
                     + 3 * (fn(s + h) + fn(s + 2 * h))
                     +      fn(s + 3 * h);
                }
                return sum / 8.0f * h * 3.0f;
            };
            auto boole_integral = [=](const float a, const float b) -> float {
                // Boole integral
                const float h = (b - a) / n;
                float sum = 0.f;
                for (int i = 0; i < n - 1; i += 4) {
                    const float s = a + h * i;
                    sum += 7 * fn(s) 
                        + 32 * fn(s + h) 
                        + 12 * fn(s + 2 * h) 
                        + 32 * fn(s + 3 * h) 
                        +  7 * fn(s + 4 * h);
                }
                return sum / 22.5f * h;
            };
            auto gauss_integral = [=](const float a, const float b) -> float {
                // Gaussian 5 point integral, seems to have bias because of rounding errors
                const float p[5] = { +std::sqrt((70.f + std::sqrt(1120.f)) / 126.f), // +-high
                                     -std::sqrt((70.f + std::sqrt(1120.f)) / 126.f),
                                     +std::sqrt((70.f - std::sqrt(1120.f)) / 126.f), // +-low
                                     -std::sqrt((70.f - std::sqrt(1120.f)) / 126.f),
                                      0.f};
                float w[5] = { 0 };
                for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < 5; j++) for (int k = 0; k < 5; k++) if (k != i && j != i && k != j) w[i] += p[j] * p[k];
                    float temp = 4.f; for (int k = 0; k < 5; k++) if (k != i) temp *= p[k];
                    w[i] = w[i] * 2.f / 3.f + temp + 0.8f;
                    for (int k = 0; k < 5; k++) if (k != i) w[i] /= (p[i] - p[k]);
                }
                const float h = (b - a) / n,  dx = (5 * h) / 2;
                float sum = 0.f;
                for (int j = 0; j < n - 1; j += 4) {
                    const float s = a + h * (j + 2.5f);
                    sum += w[0] * (fn(dx * p[0] + s)         + fn( - dx * p[1] + s + h))
                        +  w[2] * (fn(dx * p[2] + s + h * 2) + fn( - dx * p[3] + s + h * 3))
                        +  w[4] *  fn(s + h * 4);
                }
                return sum * h;
            };
            auto gauss3_integral = [=](const float a, const float b) -> float {
                // Gaussian 3 point integral
                const float p0 = -std::sqrt(3.f / 5.f), p1 =  std::sqrt(3.f / 5.f),
                            w0 = 5.f / 9.f, w1 = 8.f / 9.f,
                            h = (b - a) / n,  dx = (3 * h) / 2;
                float sum = 0.f;
                for (int j = 0; j < n - 1; j += 2) {
                    const float s = a + h * (j + 1.5f);
                    sum += w0 * (fn(dx * p0 + s) + fn(dx * p1 + s + h * 2)) + w1 * fn(s + h);
                }
                return sum * h;
            };
            auto trapezoidal_integral = [=](const float a, const float b) -> float {
                // Trapezoidal integral
                const float h = (b - a) / n;
                float sum = (fn(a) + fn(b)) * .5f;
                for (int j = 1; j < n; j++) {
                    sum += fn(a + j * h);
                }
                return sum * h;
            };
            auto midpoint_integral = [=](const float a, const float b) -> float {
                // Rectangle (Riemann) midpoint integral
                const float dx = (b - a) / n;
                float sum = 0.f;
                for (int j = 0; j < n; j++) {
                    sum += fn(a + (j + 0.5f) * dx);
                }
                return sum * dx;
            };
            const float s0 = sigma[m], s1 = sigma_reshaper(sigma[m + 1], m),
                        s  = simpson_integral    (s0, s1),
                        b  = boole_integral      (s0, s1),
                        t  = trapezoidal_integral(s0, s1),
                        m1 = midpoint_integral   (s0, s1),
                        s3 = simpson38_integral  (s0, s1),
                        g  = gauss_integral      (s0, s1),
                        g3 = gauss3_integral     (s0, s1);
            // using mix of several integrals
            return (((double)s + b + t + m1 + s3 - g) / 2 + g3) / 3;
            //return (b + s3) / 2; // needs 120k divisions
        };
#else  // ORIGINAL_SAMPLER_ALGORITHMS
        auto linear_multistep_coeff = [=](const int order, const int m, const int j) -> float {
            // using Riemann middle integral with 256k samples
            constexpr int n = 262144;
            const float a = sigma[m], dx = (sigma_reshaper(sigma[m + 1], m) - a) / n, s = sigma[m - j];
            float sum = 0.f;
            for (int h = 0; h < n; h++) {
                const float b = a + (h + 0.5f) * dx;
                float prod = 1.f;
                for (int k = 0; k < j; k++) {
                    const float t = sigma[m - k];
                    prod *= (b - t) / (s - t);
                }
                for (int k = j + 1; k < order; k++) {
                    const float t = sigma[m - k];
                    prod *= (b - t) / (s - t);
                }
                sum += prod;
            }
            return sum * dx;
        };
#endif // ORIGINAL_SAMPLER_ALGORITHMS

        // shifting 3 history elements
        if (i) for (int k = 3; k; k--) sampler_history_buffer[k] = sampler_history_buffer[k - 1];

        // computing coefficients
        for (int c = 0; c < std::min(i + 1, 4); c++)
            lms_coeff[c] =  linear_multistep_coeff(std::min(i + 1, 4), i, c);

        for (int c = 0; c < 4; c++)
        {
            ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
            float *b0_ptr = sampler_history_buffer[0].channel(c),
                  *b1_ptr = sampler_history_buffer[1].channel(c),
                  *b2_ptr = sampler_history_buffer[2].channel(c),
                  *b3_ptr = sampler_history_buffer[3].channel(c);
            for (; x_ptr < x_ptr_end; x_ptr++)
            {
                float d = (*x_ptr - *d_ptr++) / sigma[i]; *b0_ptr++ = d; // put to history
                switch (i) {
                case 0: // first step, using derivative only
                    *x_ptr += d * lms_coeff[0];
                    break;
                case 1: // second, using one history point
                    *x_ptr += d         * lms_coeff[0]
                            + *b1_ptr++ * lms_coeff[1];
                    break;
                case 2: // third, using two history points
                    *x_ptr += d         * lms_coeff[0]
                            + *b1_ptr++ * lms_coeff[1]
                            + *b2_ptr++ * lms_coeff[2];
                    break;
                default: // fourth+, using three history points
                    *x_ptr += d         * lms_coeff[0]
                            + *b1_ptr++ * lms_coeff[1]
                            + *b2_ptr++ * lms_coeff[2]
                            + *b3_ptr++ * lms_coeff[3];
                }
            }
        }
        break;
    } // lms

    case LCM: // Latent consistency models
    // adapted from https://github.com/leejet/stable-diffusion.cpp
    {
        float sigma_next = sigma[i + 1];
        if (sigma_next <= 0) {
            x_mat = denoised;
        } else {
            std::srand(seed++);
            ncnn::Mat randn = randn_4_w_h(rand() % 1000, g_main_args.m_latw, g_main_args.m_lath);
            for (int c = 0; c < 4; c++)
            {
                ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
                float* r_ptr = randn.channel(c);
                for (; x_ptr < x_ptr_end; x_ptr++)
                {
                    *x_ptr = *d_ptr + sigma_next * *r_ptr;
                    d_ptr++;
                    r_ptr++;
                }
            }
        }
        break;
    } // lcm

    default: { // Euler Ancestral
#if       ORIGINAL_SAMPLER_ALGORITHMS
    // original copy
    float sigma_up = std::min(sigma[i + 1], std::sqrt(sigma[i + 1] * sigma[i + 1] * (sigma[i] * sigma[i] - sigma[i + 1] * sigma[i + 1]) / (sigma[i] * sigma[i])));
    float sigma_down = std::sqrt(sigma[i + 1] * sigma[i + 1] - sigma_up * sigma_up);
    std::srand(seed++);
    ncnn::Mat randn = randn_4_w_h(rand() % 1000, g_main_args.m_latw, g_main_args.m_lath);

    for (int c = 0; c < 4; c++)
    {
        ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
        float* r_ptr = randn.channel(c);

        for (; x_ptr < x_ptr_end; x_ptr++)
        {
            *x_ptr = *x_ptr + ((*x_ptr - *d_ptr) / sigma[i]) * (sigma_down - sigma[i]) + *r_ptr * sigma_up;
            d_ptr++;
            r_ptr++;
        }
    }
#else  // ORIGINAL_SAMPLER_ALGORITHMS
    const double double_sigma_up = std::min((double)sigma[i + 1], 
        std::abs(sigma[i + 1] * 
                 std::sqrt(((double)sigma[i] * sigma[i] - (double)sigma[i + 1] * sigma[i + 1])) / 
                 sigma[i]));
    const float sigma_down = std::sqrt((double)sigma[i + 1] * sigma[i + 1] - double_sigma_up * double_sigma_up);
    const float sigma_up = double_sigma_up;
    std::srand(seed++);
    ncnn::Mat randn = randn_4_w_h(rand() % 1000, g_main_args.m_latw, g_main_args.m_lath);

    const float sigma_mul = sigma_down / sigma[i];
    for (int c = 0; c < 4; c++)
    {
        ONNXSTREAM_SD_INIT_X_PTR_INPUT_AND_D_PTR_DENOISED_POINTERS;
        float* r_ptr = randn.channel(c);
        for (; x_ptr < x_ptr_end; x_ptr++)
        {
            *x_ptr = (*x_ptr - *d_ptr) * sigma_mul + *d_ptr + *r_ptr * sigma_up;
            d_ptr++;
            r_ptr++;
        }
    }
#endif // ORIGINAL_SAMPLER_ALGORITHMS
    //break; // if will not be default
    } // euler_a
    } // g_main_args.sampler

    co_return 0;
} // process_sample()
