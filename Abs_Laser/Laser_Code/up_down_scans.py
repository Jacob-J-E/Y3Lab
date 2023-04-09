from funcs import *


R_1_R_2 = 0.995*0.995
F = 0.5 * (4 * (R_1_R_2)**(1/4))/(1-(R_1_R_2)**(1/2))
fsr_theoretical = (3e8/(4*20e-2))



data = pd.read_csv(r"Abs_Laser\Data\21-03-2023\ZB0.CSV")
data_DB_free = pd.read_csv(r"Abs_Laser\Data\21-03-2023\Z0.CSV")

x_axis = data_DB_free['in s']
c1 = data_DB_free['C1 in V']           
c2 = data_DB_free['C2 in V'] 
c3 = data_DB_free['C3 in V']
c4 = data_DB_free['C4 in V']
c1_B = data['C1 in V']

c1_B = c1_B/max(c1_B)*max(c1)

c1_B = np.array(c1_B)
c3 = np.array(c3)
c4 = 100*np.array(c4)
c1 = np.array(c1)
x_axis = np.array(x_axis)


#SPLIT DATA INTO UP/DOWN -------------------------------------------------------------------------------

# c1_first = c1[:round(len(x_axis)/2)]
# x_first = x_axis[:round(len(x_axis)/2)]
# fp_first = c4[:round(len(x_axis)/2)]
# c1B_first = c1_B[:round(len(x_axis)/2)]

# c1_first = c1_first[x_first>-0.01286]
# fp_first = fp_first[x_first>-0.01286]
# c1B_first = c1B_first[x_first>-0.01286]
# x_first = x_first[x_first>-0.01286]

c1_first = c1[(x_axis<0.0220) & (x_axis > -0.00799)]
fp_first = c4[(x_axis<0.0220) & (x_axis > -0.00799)]
c1B_first = c1_B[(x_axis<0.0220) & (x_axis > -0.00799)]
x_first = x_axis[(x_axis<0.0220) & (x_axis > -0.00799)]

# c1_second = c1[round(len(x_axis)/2):]
# x_second = x_axis[round(len(x_axis)/2):]
# fp_second = c4[round(len(x_axis)/2):]
# c1B_second = c1_B[round(len(x_axis)/2):]


# c1_second = c1_second[x_second<0.06060]
# fp_second = fp_second[x_second<0.06060]
# c1B_second = c1B_second[x_second<0.06060]
# x_second = x_second[x_second<0.06060]

c1_second = c1[x_axis>0.0225]
fp_second = c4[x_axis>0.0225]
c1B_second = c1_B[x_axis>0.0225]
x_second = x_axis[x_axis>0.0225]

# Plots to check the splitting worked-------------------------------------------------------------------------------

plt.plot(x_first,c1_first,color='black',label="First")
plt.plot(x_first,fp_first,color='black')
plt.plot(x_second,c1_second,color='red',label="Second")
plt.plot(x_second,fp_second,color='red')
# plt.legend(loc='upper right')
plt.show()

iterations = 100
max_poly_order = 4
min_poly_order = 2
if min_poly_order > max_poly_order:
    print('Make sure max poly order is higher or equal to min poly order')
    exit()
elif min_poly_order < 2:
    print('Please put a coeff greater than or equal to 2')
    exit()





x_axis = x_first
c4 = fp_first
c1 = c1_first
c1_B = c1B_first

x_axis = x_axis[::-1]
x_axis, c4,c1,c1_B = zip(*sorted(zip(x_axis, c4,c1,c1_B )))

c1_B = np.array(c1_B)
c3 = np.array(c3)
c4 = np.array(c4)
c1 = np.array(c1)
x_axis = np.array(x_axis)

#NORMALIZING X AXIS -------------------------------------------------------------------------------
length_of_xaxis = len(x_axis)
normalized_x_axis = []
for i in range(len(x_axis)):
    normalized_x_axis.append((i-(length_of_xaxis/2))/(length_of_xaxis/2))

normalized_x_axis = np.array(normalized_x_axis)

#NORMALIZING Y AXIS FP-------------------------------------------------------------------------------
c4 = c4 - min(c4)
c4 = c4/max(c4)
#FITTING LORENZIANS TO FP AND FINDING PEAK VALUES----------------------------------------------------
# c4_copy = c4
# c4_copy = c4_copy - min(c4_copy)
# c4_copy = c4_copy/max(c4_copy)


peaks, _= find_peaks(c4, distance=1000)
c4_peaks = c4[peaks]
x_axis_peaks = normalized_x_axis[peaks]
x_axis_peaks = x_axis_peaks[1:]
c4_peaks = c4_peaks[1:]
spacing = np.diff(x_axis_peaks)
plt.scatter(x_axis_peaks,c4_peaks)
plt.plot(normalized_x_axis,c4)
plt.show()

lorentzian_array = np.zeros(len(normalized_x_axis))
FP_lorenzian_x_axis_peaks = []
FP_lorenzian_x_axis_peaks_amplitude = []
for i in range(len(x_axis_peaks)):
    inital_guess = [x_axis_peaks[i], 0.00004,c4_peaks[i], 0.2]  
    para, cov = curve_fit(lorentzian,normalized_x_axis, c4, inital_guess)
    # y_data = lorentzian(normalized_x_axis,para[0], para[1], para[2], para[3])
    y_data = lorentzian(normalized_x_axis,para[0], para[1], 1*para[1], para[3])

    plt.plot(normalized_x_axis,y_data)

    FP_lorenzian_x_axis_peaks.append(para[0])
    FP_lorenzian_x_axis_peaks_amplitude.append(max(y_data) - para[3])

    lorentzian_array += y_data - para[3]
    # print(f"({para[0]},{max(y_data) - para[3]})")
FP_lorenzian_x_axis_peaks = np.array(FP_lorenzian_x_axis_peaks)
FP_lorenzian_x_axis_peaks_amplitude = np.array(FP_lorenzian_x_axis_peaks_amplitude)
plt.scatter(x_axis_peaks,c4_peaks)
plt.plot(normalized_x_axis,c4)
plt.show()


#FINDING A_0 AND A_1 ----------------------------------------------------------------------------

relative_FSP_SHIFT = 0
relative_FSP_SHIFT_ARRAY = []
for i in range(len(FP_lorenzian_x_axis_peaks)):
    relative_FSP_SHIFT_ARRAY.append(relative_FSP_SHIFT)
    relative_FSP_SHIFT += fsr_theoretical

relative_FSP_SHIFT_ARRAY = np.array(relative_FSP_SHIFT_ARRAY)

m = (relative_FSP_SHIFT_ARRAY[-1] - relative_FSP_SHIFT_ARRAY[0])/ (FP_lorenzian_x_axis_peaks[-1] - FP_lorenzian_x_axis_peaks[0])
c = relative_FSP_SHIFT_ARRAY[0] - m*FP_lorenzian_x_axis_peaks[0]
inital_guess_straight = [m,c]
para_straight, cov_straight = curve_fit(straight_line,FP_lorenzian_x_axis_peaks,relative_FSP_SHIFT_ARRAY,inital_guess_straight)

print('***GUESS FOR A_1 AND A_0****')
print(f'Fitted Gradient (A_1): {para_straight[0]:.4g} +/- {cov_straight[0][0]**(0.5):.4g}')
print(f'Fitted Intercept (A_0): {para_straight[1]:.4g} +/- {cov_straight[1][1]**(0.5):.4g}')
print('****************************')

a_0_guess = para_straight[1]
a_1_guess = para_straight[0]

plt.scatter(FP_lorenzian_x_axis_peaks,relative_FSP_SHIFT_ARRAY)
plt.plot(FP_lorenzian_x_axis_peaks,straight_line(FP_lorenzian_x_axis_peaks,a_0_guess,a_1_guess),color='black')


# relative_FSP_SHIFT_ARRAY = np.array(relative_FSP_SHIFT_ARRAY)
# FP_lorenzian_x_axis_peaks = np.array(FP_lorenzian_x_axis_peaks)

grad_val = (relative_FSP_SHIFT_ARRAY[-1]-relative_FSP_SHIFT_ARRAY[0])/(FP_lorenzian_x_axis_peaks[-1]-FP_lorenzian_x_axis_peaks[0])
c_val = relative_FSP_SHIFT_ARRAY[0]-grad_val*relative_FSP_SHIFT_ARRAY[0]

plt.plot(FP_lorenzian_x_axis_peaks,straight_line(FP_lorenzian_x_axis_peaks,grad_val,a_0_guess),color='red')
plt.show()

# a_1_guess = grad_val

#Plotting modified airy function ----------------------------------------------------------------

plt.scatter(normalized_x_axis,lorentzian_array)
plt.show()
array_of_coeffients = []
b_values = []
for i in range(min_poly_order,max_poly_order+1):
    poly_order = i 
    # ordering -> a_0,a_1, a_2, a_3, a_4, b_0, b_1)
    inital_guess_airy_modified = [a_0_guess,a_1_guess]

    for i in range(poly_order+1):
        inital_guess_airy_modified.append(0)

    with alive_bar(iterations) as bar:
        for i in range(iterations):
            bar()
            #bounds= ((0,0,0,0,0,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf))
            para_airy_modified, cov_airy_modified = curve_fit(airy_modified_function,normalized_x_axis,lorentzian_array,inital_guess_airy_modified)
            inital_guess_airy_modified = para_airy_modified

    a_coeffients = para_airy_modified[:poly_order+1]
    b_coeff = [para_airy_modified[-2],para_airy_modified[-1]]
    array_of_coeffients.append(a_coeffients)
    b_values.append(b_coeff)
    print('****************************')
    print(f'COMPLETED BOOTSTRAPING with Poly Order {poly_order}')
    print(f'Poly Order {poly_order} - a coeffients {a_coeffients}')
    print(f'Poly Order {poly_order} - b coeffients {b_coeff}')
    print('****************************')


plt.figure()
domain_airy_modified = np.linspace(min(normalized_x_axis),max(normalized_x_axis), 100000)

plt.title('Fitted FP using Modified Airy Function')
for i in range(len(array_of_coeffients)):
    para = list(array_of_coeffients[i]) + list(b_values[i])
    plt.plot(domain_airy_modified,airy_modified_function(domain_airy_modified,*para), label =  f"Fitted Airy (f(x) poly order {min_poly_order + i})" )
plt.plot(normalized_x_axis,lorentzian_array, label =  "Data" )
plt.legend()

plt.figure()
plt.title('Frequency Scaled FP')
for i in range(len(array_of_coeffients)):
    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
    plt.plot(freqency_array,lorentzian_array, label =  f"Frequency Scaling (poly order {min_poly_order + i})" )
plt.xticks(np.arange(0, max(freqency_array)+fsr_theoretical, fsr_theoretical))
plt.legend()



plt.figure()
plt.title('Frequency Scaled FP Peak Spacings')
colors = plt.cm.rainbow(np.linspace(0, 1, len(array_of_coeffients)))
for i in range(len(array_of_coeffients)):
    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
    freq_peaks = freqency_array[peaks]
    freq_peaks = freq_peaks[1:]
    mid_val = (freq_peaks[1:] + freq_peaks[:-1])/2
    spacing_freq = np.diff(freq_peaks)
    plt.scatter(mid_val,spacing_freq, label = f'Frequency FP spacing difference ({(max(spacing_freq) - min(spacing_freq)):.4g})[poly order {min_poly_order + i}]', color = colors[i])
    plt.axhline(max(spacing_freq), color = colors[i])
    plt.axhline(min(spacing_freq), color = colors[i])
plt.legend()

plt.figure()
plt.title('Fitted FP Lorentzian (Original Dataset) Peak Spacings')
mid_point = (x_axis_peaks[1:] + x_axis_peaks[:-1]) / 2
plt.scatter(mid_point,spacing, label = 'Time scale')
plt.legend()


plt.show()
#Plotting the Broadened Absorption Spectra----------------------------------------------------------------
plt.figure()
plt.title('Frequency Scaled Broadened Absorption Spectra')
for i in range(len(array_of_coeffients)):
    peaks_fine, _= find_peaks(-1*c1_B, distance=8000)
    c1_b_grouped_peaks =c1_B[peaks_fine]
    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
    freq_peaks = freqency_array[peaks_fine]

    centers_fine = [freq_peaks[1],freq_peaks[2],freq_peaks[4],freq_peaks[6]]
    amplitude_fine = [c1_b_grouped_peaks[1],c1_b_grouped_peaks[2],c1_b_grouped_peaks[4],c1_b_grouped_peaks[6]]

    Rb_85_Ground_State_Difference = (freq_peaks[4]-freq_peaks[2])/1e9
    Rb_87_Ground_State_Difference = (freq_peaks[6]-freq_peaks[1])/1e9

    Rb_85_Ground_State_Difference_THEORY = 3.0357324390
    Rb_87_Ground_State_Difference_THEORY = 6.834682610904290

    Rb_85_Ground_State_Difference_THEORY_PD = ((Rb_85_Ground_State_Difference_THEORY - Rb_85_Ground_State_Difference) / Rb_85_Ground_State_Difference_THEORY) * 100
    Rb_87_Ground_State_Difference_THEORY_PD = ((Rb_87_Ground_State_Difference_THEORY - Rb_87_Ground_State_Difference) / Rb_87_Ground_State_Difference_THEORY) * 100
  
    print('************************************')
    print(f'Poly Order: {min_poly_order + i}')
    print(f'Hyperfine Structure Rb 85 Ground State Difference: {Rb_85_Ground_State_Difference:.4g} MHz, Theory P.D: {Rb_85_Ground_State_Difference_THEORY_PD:.4g}%')
    print(f'Hyperfine Structure Rb 87 Ground State Difference: {Rb_87_Ground_State_Difference:.4g} MHz, Theory P.D: {Rb_87_Ground_State_Difference_THEORY_PD:.4g}%')
    print('****************************')

    plt.plot(freqency_array,c1_B, label = f'Data (poly order {min_poly_order + i})')
    plt.scatter(centers_fine,amplitude_fine, marker= 'x', label = f'Scatter (poly order {min_poly_order + i})')
plt.legend()

#Plotting Hyperfine Structure ------------------------------------------------------
plt.figure()
plt.title('Frequency Scaled Hyperfine Structure')
for i in range(len(array_of_coeffients)):
    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
    hyper_fine_structure = c1-c1_B
    hyper_fine_structure = hyper_fine_structure - min(hyper_fine_structure)
    hyper_fine_structure = hyper_fine_structure/max(hyper_fine_structure)
    plt.plot(freqency_array,hyper_fine_structure,label = f'Data Hyperfine Structure (poly order {min_poly_order + i})')
plt.legend()


plt.figure()
plt.title('Frequency Scaled Hyperfine Structure (Rb 87 Ground State F = 2 -> F=1,2,3)')
# for i in range(len(array_of_coeffients)):
#     freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
#     hyper_fine_structure = c1-c1_B

#     hyper_fine_structure = hyper_fine_structure[(freqency_array > 6.954e8) & (freqency_array < 1.4992e9)]
#     freqency_array = freqency_array[(freqency_array > 6.954e8) & (freqency_array < 1.4992e9)]

#     hyper_fine_structure = hyper_fine_structure - min(hyper_fine_structure)
#     hyper_fine_structure = hyper_fine_structure/max(hyper_fine_structure)

#     plt.plot(freqency_array,hyper_fine_structure,label = f'Data Hyperfine Structure (poly order {min_poly_order + i})')

#     peaks_fine, _= find_peaks(hyper_fine_structure, distance=400)
#     subtracted_peaks = hyper_fine_structure[peaks_fine]
#     freq_peaks = freqency_array[peaks_fine]

#     peaks_values = [subtracted_peaks[2],subtracted_peaks[3],subtracted_peaks[4],subtracted_peaks[5],subtracted_peaks[7],subtracted_peaks[1]]
#     freq_values = [freq_peaks[2],freq_peaks[3],freq_peaks[4],freq_peaks[5],freq_peaks[7],freq_peaks[1]]
#     peaks_values_inital_guess = np.array(peaks_values) * 1e7
#     width = 9e6
#     #initial_guess = [freq_values[0],width,peaks_values_inital_guess[0],freq_values[1],width,peaks_values_inital_guess[1],freq_values[2],width,peaks_values_inital_guess[2],freq_values[3],width,peaks_values_inital_guess[3],freq_values[4],width,peaks_values_inital_guess[4], 1e-30,1e-30,0.03]
#     #f1,w1,a1,c1,f2,w2,a2,c2,f3,w3,a3,c3,f4,w4,a4,c4,f5,w5,a5,c5,c

#     m = (hyper_fine_structure[-1] - hyper_fine_structure[0])/ (freqency_array[-1] - freqency_array[0])
#     c = hyper_fine_structure[0] - m*freqency_array[0]
#     maybe_peak = 2*freq_values[0] - freq_values[1]
#     initial_guess = [freq_values[0],width+8e6,peaks_values_inital_guess[0],freq_values[1],width+30e6,peaks_values_inital_guess[1]+0.8e6,freq_values[2],width,peaks_values_inital_guess[2],freq_values[3],width,peaks_values_inital_guess[3],freq_values[4],width,peaks_values_inital_guess[4],maybe_peak,width,peaks_values_inital_guess[5],m,c]
#     params, cov = curve_fit(five_lor_x_update,freqency_array,hyper_fine_structure,initial_guess)
#     plt.plot(freqency_array,five_lor_x_update(freqency_array,*params),color='black')
#     # plt.plot(freqency_array,five_lor_x_update(freqency_array,*params) - np.array((params[-3]*freqency_array**2 + params[-2]*freqency_array - params[-1])),color='red')
#     plt.plot(freqency_array,five_lor_x_update(freqency_array,*initial_guess),label = 'Inital Guess', color = 'orange')
#     plt.plot(freqency_array,hyper_fine_structure - straight_line(freqency_array,initial_guess[-2],initial_guess[-1]), label = 'SHIFTED', color = 'purple')
#     print((freq_values[4] - freq_values[1])/1e6)
#     print((freq_values[1] - maybe_peak)/1e6)


#     Rb_87_2_to_3_THEORY = 266.6500
#     Rb_87_1_to_2_THEORY = 156.9470

#     direct_Rb_87_2_to_3 = (params[12] - params[3])/ 1E6
#     direct_Rb_87_2_to_3_Difference_THEORY_PD = 100* ((direct_Rb_87_2_to_3 - Rb_87_2_to_3_THEORY) / Rb_87_2_to_3_THEORY)

#     c1_2 = params[0]
#     c1_3 = params[6]
#     c2_3 = params[9]

#     # print(indirect_Rb_87_g2_to_1, 'MONEY')
#     # print(indirect_Rb_87_g2_to_3, 'MONEY')
#     # print(indirect_Rb_87_g2_to_2, 'MONEY')

#     indirect_Rb_87_1_to_2 = 2*(c2_3 - c1_3)/1e6
#     indirect_Rb_87_2_to_3 = 2*(c1_3 - c1_2)/1e6

#     indirect_Rb_87_1_to_2_Difference_THEORY_PD = 100*((indirect_Rb_87_1_to_2 - Rb_87_1_to_2_THEORY)/Rb_87_1_to_2_THEORY)
#     indirect_Rb_87_2_to_3_Difference_THEORY_PD = 100*((indirect_Rb_87_2_to_3 - Rb_87_2_to_3_THEORY)/Rb_87_2_to_3_THEORY)

#     print('************************************')
#     print(f'Poly Order: {min_poly_order + i}')
#     print(f'(Direct Method) Hyperfine Structure Rb 87 excited - F=2 -> 3 : {direct_Rb_87_2_to_3:.4g} MHz, Theory P.D: {direct_Rb_87_2_to_3_Difference_THEORY_PD:.4g}%')
#     print(f'(Indirect Method) Hyperfine Structure Rb 87 excited - F=2 -> 3 : {indirect_Rb_87_2_to_3:.4g} MHz, Theory P.D: {indirect_Rb_87_2_to_3_Difference_THEORY_PD:.4g}%')
#     print(f'(Indirect Method) Hyperfine Structure Rb 87 excited - F=1 -> 2 : {indirect_Rb_87_1_to_2:.4g} MHz, Theory P.D: {indirect_Rb_87_1_to_2_Difference_THEORY_PD:.4g}%')
#     print('************************************')
#     plt.scatter(freq_peaks,subtracted_peaks, marker= 'x',label = f'Peak Points (poly order {min_poly_order + i})')
# plt.legend()

# plt.figure()
# plt.title('Frequency Scaled Hyperfine Structure (Rb 85 Ground State F = 3 -> F=2,3,4)')
# for i in range(len(array_of_coeffients)):
#     freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
#     hyper_fine_structure = c1-c1_B

#     hyper_fine_structure = hyper_fine_structure[(freqency_array > 2.1e9) & (freqency_array < 2.6e9)]
#     freqency_array = freqency_array[(freqency_array > 2.1e9) & (freqency_array < 2.6e9)]

#     hyper_fine_structure = hyper_fine_structure - min(hyper_fine_structure)
#     hyper_fine_structure = hyper_fine_structure/max(hyper_fine_structure)

#     plt.plot(freqency_array,hyper_fine_structure,label = f'Data Hypserfine Structure (poly order {min_poly_order + i})')

#     peaks_fine, _= find_peaks(hyper_fine_structure, distance=150)
#     subtracted_peaks = hyper_fine_structure[peaks_fine]
#     freq_peaks = freqency_array[peaks_fine]

#     peaks_values = [subtracted_peaks[2],subtracted_peaks[5],subtracted_peaks[6],subtracted_peaks[8]]
#     freq_values = [freq_peaks[2],freq_peaks[5],freq_peaks[6],freq_peaks[8]]
#     peaks_values_inital_guess = np.array(peaks_values)
#     m = (hyper_fine_structure[-1] - hyper_fine_structure[0])/ (freqency_array[-1] - freqency_array[0])
#     c = hyper_fine_structure[0] - m*freqency_array[0]
#     #initial_guess = [freq_values[0]-0.005e9,width+30e6,peaks_values_inital_guess[0]+3e6,freq_values[1],width+20e6,peaks_values_inital_guess[1]+30e6,freq_values[2],width,peaks_values_inital_guess[2]+3e6,freq_values[3],width,peaks_values_inital_guess[3],m,c]
#     initial_guess = [freq_values[0],width,peaks_values_inital_guess[0],freq_values[1],width,peaks_values_inital_guess[1],freq_values[2],width,peaks_values_inital_guess[2],freq_values[3],width,peaks_values_inital_guess[3],m,c]
#     params, cov = curve_fit(four_lor_x_update,freqency_array,hyper_fine_structure,initial_guess)
#     plt.plot(freqency_array,four_lor_x_update(freqency_array,*params),color='black')
#     plt.plot(freqency_array,four_lor_x_update(freqency_array,*initial_guess),label = 'Inital Guess', color = 'orange')
#     plt.plot(freqency_array,hyper_fine_structure - straight_line(freqency_array,initial_guess[-2],initial_guess[-1]), label = 'SHIFTED', color = 'purple')
#     plt.scatter(freq_values,peaks_values, marker= 'x',label = f'Peak Points (poly order {min_poly_order + i})')
#     #plt.scatter(freq_peaks,subtracted_peaks, marker= 'x', color = 'red',label = f'Peak Points (poly order {min_poly_order + i})')
#     Rb_85_2_to_3_THEORY = 63.401
#     Rb_85_3_to_4_THEORY = 120.640

#     c_2_4 = params[3]
#     c_3_4 = params[6]
#     v_4 = params[9]
    

#     indirect_Rb_85_2_to_3 = 2*(c_3_4 - c_2_4)/1e6
#     indirect_Rb_85_3_to_4 = 2*(v_4 - c_3_4)/1e6

#     indirect_Rb_85_2_to_3_Difference_THEORY_PD = 100*((indirect_Rb_85_2_to_3 - Rb_85_2_to_3_THEORY)/Rb_85_2_to_3_THEORY)
#     indirect_Rb_85_3_to_4_Difference_THEORY_PD = 100*((indirect_Rb_85_3_to_4 - Rb_85_3_to_4_THEORY)/Rb_85_3_to_4_THEORY)

#     print('************************************')
#     print(f'Poly Order: {min_poly_order + i}')
#     print(f'(Indirect Method) Hyperfine Structure Rb 85 excited - F=2 -> 3 : {indirect_Rb_85_2_to_3:.4g} MHz, Theory P.D: {indirect_Rb_85_2_to_3_Difference_THEORY_PD:.4g}%')
#     print(f'(Indirect Method) Hyperfine Structure Rb 85 excited - F=3 -> 4 : {indirect_Rb_85_3_to_4:.4g} MHz, Theory P.D: {indirect_Rb_85_3_to_4_Difference_THEORY_PD:.4g}%')
#     print('************************************')

plt.legend()

plt.show()

